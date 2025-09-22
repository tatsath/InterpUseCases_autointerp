# hallu_probe.py
import os, json, random, re
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
import joblib

# -----------------------------
# 0) Config
# -----------------------------
model_id = "meta-llama/Llama-2-7b-hf"   # <-- replace with your 7B model path
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Which layer/feature to probe
LAYER_IDX = 20          # mid-high layer often works well; tune 12..28 for 7B/32L
POOLING = "last_gen"    # "last_gen" | "mean_gen" | "cls_like" (first token)
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.8        # higher temp => more hallucinations (for training diversity)

# -----------------------------
# 1) Minimal data interface
# -----------------------------
# Import financial prompts
from financial_data import financial_prompts

# Use financial domain prompts for hallucination detection
train_items = financial_prompts

# -----------------------------
# 2) Load model/tokenizer
# -----------------------------
tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map="auto",          # works well on H100s
)
model.eval()

gen_cfg = GenerationConfig(
    do_sample=True,
    temperature=TEMPERATURE,
    top_p=0.9,
    max_new_tokens=MAX_NEW_TOKENS
)

# -----------------------------
# 3) Helpers
# -----------------------------
def simple_prompt(q: str) -> str:
    # Keep it plain for base models; adapt for chat templates if using -Instruct
    return f"Q: {q}\nA:"

def generate_answer(question: str) -> str:
    prompt = simple_prompt(question)
    enc = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(**enc, generation_config=gen_cfg)
    text = tok.decode(out_ids[0], skip_special_tokens=True)
    # Extract answer after "A:" if present
    ans = text.split("A:")[-1].strip()
    return ans

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-\.,']", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def label_hallucination(pred: str, gold: str) -> int:
    """
    Returns 1 if hallucinated (incorrect), 0 if correct.
    Simple heuristic: substring containment or fuzzy exactness.
    Replace with your grader if you want stricter labels.
    """
    p = normalize_text(pred)
    g = normalize_text(gold)
    if g in p or p in g:
        return 0
    # loose containments / common aliases could be added here
    return 1

@torch.no_grad()
def extract_features(question: str, answer: str, layer_idx=LAYER_IDX, pooling=POOLING) -> np.ndarray:
    """
    Re-encode (prompt + generated answer) in one forward pass and grab hidden states.
    This approximates the generation-time residuals closely and is simple/fast.
    """
    full = f"{simple_prompt(question)} {answer}"
    enc = tok(full, return_tensors="pt", truncation=True, max_length=2048).to(device)
    # We need hidden states for the selected layer:
    outputs = model.model(
        **{k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")},
        output_hidden_states=True
    )
    # For LLaMA in HF, embeddings are in outputs.hidden_states; last is final layer.
    # hidden_states: tuple(len = n_layers+1)[batch, seq, hidden]
    H = outputs.hidden_states[layer_idx]  # [1, T, d]
    T = H.shape[1]

    if pooling == "last_gen":
        # assume the *last token* corresponds mainly to generated tail
        vec = H[0, -1, :].float().cpu().numpy()
    elif pooling == "mean_gen":
        # mean over last k tokens (up to 64 or up to the answer length)
        k = min(32, T)  # you can tune this window
        vec = H[0, T-k:T, :].mean(dim=0).float().cpu().numpy()
    elif pooling == "cls_like":
        # first token (not perfect CLS for decoder-only, but useful baseline)
        vec = H[0, 0, :].float().cpu().numpy()
    else:
        raise ValueError("Unknown POOLING")
    return vec

# -----------------------------
# 4) Build training set
# -----------------------------
X, y = [], []
for item in train_items:
    q, gold = item["question"], item["answer"]
    pred = generate_answer(q)
    lab = label_hallucination(pred, gold)
    feat = extract_features(q, pred, layer_idx=LAYER_IDX, pooling=POOLING)
    X.append(feat); y.append(lab)

X = np.vstack(X).astype(np.float32)
y = np.array(y, dtype=np.int64)

# Split for quick sanity check (without stratification due to class imbalance)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 5) Train linear probe
# -----------------------------
probe = LogisticRegression(
    penalty="l2",
    C=1.0,
    max_iter=2000,
    class_weight="balanced",   # helps if positives/negatives imbalanced
    n_jobs=-1
)
probe.fit(Xtr, ytr)

# Evaluate
proba = probe.predict_proba(Xte)[:, 1]
yp = (proba >= 0.5).astype(int)
acc = accuracy_score(yte, yp)
auc = roc_auc_score(yte, proba)
p, r, f1, _ = precision_recall_fscore_support(yte, yp, average="binary")
print(f"[Probe @ layer {LAYER_IDX} / {POOLING}]  Acc={acc:.3f}  AUC={auc:.3f}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}")

# Save the probe
os.makedirs("probe_ckpt", exist_ok=True)
joblib.dump({"clf": probe, "layer_idx": LAYER_IDX, "pooling": POOLING, "model_id": model_id}, "probe_ckpt/hallu_linear_probe.joblib")

# -----------------------------
# 6) Inference helper (single prompt)
# -----------------------------
def hallu_score(prompt: str) -> Tuple[str, float]:
    """Returns (model_answer, hallucination_probability)"""
    pred = generate_answer(prompt)
    feat = extract_features(prompt, pred, layer_idx=LAYER_IDX, pooling=POOLING)
    ckpt = joblib.load("probe_ckpt/hallu_linear_probe.joblib")
    score = ckpt["clf"].predict_proba(feat.reshape(1,-1))[0,1]
    return pred, float(score)

# Quick demo
test_q = "Who painted the Mona Lisa?"
ans, score = hallu_score(test_q)
print(f"Q: {test_q}\nA: {ans}\nHallucination probability: {score:.3f}")
