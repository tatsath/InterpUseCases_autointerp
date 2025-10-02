# -*- coding: utf-8 -*-
"""
AutoInterp++: Delphi-free auto-interpretability pipeline
Complete end-to-end feature discovery, labeling, and evaluation system

Key Features:
- Domain-specific feature discovery with coverage/lift ranking
- Contrastive LLM explanation (positives vs hard negatives)
- Clustering & polysemanticity analysis (SBERT + HDBSCAN)
- Thresholding with selectivity gate
- Full metrics: F1/precision/recall, selectivity, confusion matrix, PR curve
- Robustness testing (fuzzing) and self-consistency validation
- Feature Card JSON artifacts for audit-ready documentation

Usage:
1. Implement call_llm() with your LLM provider
2. Prepare your data: positives, negatives, background corpus
3. Run build_feature_card() for each latent feature
4. Get comprehensive Feature Card JSONs with all metrics
"""

import os, re, math, json, orjson, random, pathlib, itertools, hashlib
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Core dependencies
from sentence_transformers import SentenceTransformer
import faiss, hdbscan
from sklearn.metrics import silhouette_score, precision_recall_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from rapidfuzz import fuzz

# Optional: For Meta 7B/8B models
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# ------------------------------
# Utilities
# ------------------------------

def set_seed(seed: int = 1337):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TRANSFORMERS:
        torch.manual_seed(seed)

def sha(s: str) -> str:
    """Generate short hash for versioning"""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]

def write_json(obj: Any, path: str):
    """Write JSON with proper directory creation"""
    pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(orjson.dumps(obj, option=orjson.OPT_INDENT_2))

# ------------------------------
# Embeddings & FAISS for hard negatives
# ------------------------------

class EmbIndex:
    """FAISS-based embedding index for hard negative mining"""
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.matrix = None
        self.texts = None

    def build(self, texts: List[str]):
        """Build FAISS index from texts"""
        X = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=512)
        index = faiss.IndexFlatIP(X.shape[1])
        index.add(X.astype(np.float32))
        self.index, self.matrix, self.texts = index, X, texts

    def knn(self, queries: List[str], k=5) -> List[List[str]]:
        """Find k nearest neighbors for queries"""
        Q = self.model.encode(queries, convert_to_numpy=True, normalize_embeddings=True, batch_size=256)
        sims, idx = self.index.search(Q.astype(np.float32), k)
        return [[self.texts[i] for i in row] for row in idx]

# ------------------------------
# Candidate ranking (coverage, lift)
# ------------------------------

def prevalence(activations: np.ndarray, tau: float) -> float:
    """Calculate prevalence at threshold tau"""
    return float((activations >= tau).sum()) / max(1, activations.shape[0])

def coverage(token_spans_fired: int, total_tokens: int) -> float:
    """Calculate coverage metric"""
    return token_spans_fired / max(1.0, float(total_tokens))

def rank_candidates(latent_stats: Dict[int, Dict[str, float]], top_frac=0.1) -> List[int]:
    """
    Rank candidates by lift * coverage
    
    Args:
        latent_stats: Dict mapping latent_id to stats
        top_frac: Fraction of top candidates to return
    
    Returns:
        List of top latent IDs
    """
    scored = [(lid, v["lift"] * v["coverage"]) for lid, v in latent_stats.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    k = max(1, int(len(scored) * top_frac))
    return [lid for lid, _ in scored[:k]]

# ------------------------------
# Clustering & polysemanticity analysis
# ------------------------------

def cluster_exemplars(texts: List[str], 
                     embed_model="sentence-transformers/all-MiniLM-L6-v2",
                     min_cluster_size=6, 
                     min_samples=1) -> Dict[str, Any]:
    """Cluster exemplars using HDBSCAN"""
    if len(texts) < max(min_cluster_size, 8):
        return {"n_clusters": 1, "labels": [0]*len(texts), "silhouette": 0.0}
    
    model = SentenceTransformer(embed_model)
    X = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=128)
    cl = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = cl.fit_predict(X)
    mask = labels != -1
    sil = silhouette_score(X[mask], labels[mask]) if mask.sum() >= 10 and np.unique(labels[mask]).size > 1 else 0.0
    n_clusters = len(set(labels) - {-1}) or 1
    return {"n_clusters": int(n_clusters), "labels": labels.tolist(), "silhouette": float(sil)}

def polysemanticity_index(cluster_result: Dict[str, Any]) -> float:
    """Calculate polysemanticity index from clustering results"""
    return float(cluster_result["n_clusters"]) * max(0.0, cluster_result["silhouette"])

# ------------------------------
# LLM Integration (Meta 7B/8B models)
# ------------------------------

class MetaLLMExplainer:
    """LLM explainer using Meta models (7B/8B parameters)"""
    
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf", device="auto"):
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library required for Meta models")
        
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device
        )
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str, max_length=512, temperature=0.7) -> str:
        """Generate response from model"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt from response
        if prompt in response:
            response = response.replace(prompt, "").strip()
        return response

# ------------------------------
# Contrastive explanation system
# ------------------------------

CONTRASTIVE_TEMPLATE = """You are labeling a latent feature in an LLM from POSITIVE (fires) vs NEGATIVE (near-miss) text.
Return STRICT JSON with fields: label, definition, include_cues, exclude_cues, risks.
Rules:
- label: SHORT noun_phrase (e.g., "sports_achievement", "tech_innovation", "medical_condition").
- definition: one sentence, neutral.
- include_cues: 5-12 phrases/tokens that indicate positives.
- exclude_cues: 3-10 phrases/tokens that often appear in negatives.
- risks: 3-8 short notes about common false positives to avoid.

POSITIVE:
{positives}

NEGATIVE:
{negatives}

JSON Response:"""

def make_contrastive_prompt(pos: List[str], neg: List[str]) -> str:
    """Create contrastive prompt for LLM"""
    P = "\n".join(f"- {p}" for p in pos[:8])
    N = "\n".join(f"- {n}" for n in neg[:8])
    return CONTRASTIVE_TEMPLATE.format(positives=P, negatives=N)

def parse_llm_response(response: str) -> Dict[str, Any]:
    """Parse LLM response and extract JSON"""
    try:
        # Try to find JSON in response
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            return json.loads(json_str)
    except:
        pass
    
    # Fallback: return default structure
    return {
        "label": "unlabeled_feature",
        "definition": "Feature requires manual review",
        "include_cues": [],
        "exclude_cues": [],
        "risks": ["Manual review required"]
    }

def call_llm(prompt: str, llm_explainer: Optional[MetaLLMExplainer] = None) -> Dict[str, Any]:
    """
    Call LLM for explanation generation
    
    Args:
        prompt: Input prompt
        llm_explainer: Optional MetaLLMExplainer instance
    
    Returns:
        Parsed JSON response
    """
    if llm_explainer:
        response = llm_explainer.generate(prompt)
        return parse_llm_response(response)
    else:
        # Fallback: return mock response for testing
        return {
            "label": "financial_performance",
            "definition": "Feature detects financial performance indicators and metrics",
            "include_cues": ["revenue", "profit", "earnings", "growth", "performance", "financial"],
            "exclude_cues": ["loss", "decline", "negative", "decrease"],
            "risks": ["May trigger on non-financial performance", "Could miss industry-specific terms"]
        }

def explain_contrastive(positives: List[str], 
                       negatives: List[str], 
                       n_trials=3, 
                       seed=1337,
                       llm_explainer: Optional[MetaLLMExplainer] = None) -> Dict[str, Any]:
    """Generate contrastive explanation with self-consistency"""
    set_seed(seed)
    outputs = []
    
    for t in range(n_trials):
        random.shuffle(positives)
        random.shuffle(negatives)
        prompt = make_contrastive_prompt(positives, negatives)
        j = call_llm(prompt, llm_explainer)
        outputs.append(j)
    
    # Self-consistency vote on label; union cues
    label_counts = Counter([o.get("label", "") for o in outputs])
    label, votes = label_counts.most_common(1)[0]
    
    include_cues = sorted({c.strip().lower() for o in outputs for c in o.get("include_cues", []) if c.strip()})
    exclude_cues = sorted({c.strip().lower() for o in outputs for c in o.get("exclude_cues", []) if c.strip()})
    risks = sorted({c.strip() for o in outputs for c in o.get("risks", []) if c.strip()})
    definition = outputs[0].get("definition", "")
    
    return {
        "label": label or "unlabeled_feature",
        "definition": definition,
        "include_cues": include_cues[:16],
        "exclude_cues": exclude_cues[:16],
        "risks": risks[:12],
        "self_consistency": {
            "trials": n_trials,
            "top_label_votes": int(votes),
            "agreement": float(votes) / float(n_trials)
        }
    }

# ------------------------------
# Scoring systems
# ------------------------------

def compile_cues(cues: List[str]):
    """Compile cues into regex pattern"""
    safe = [re.escape(c) for c in cues if c.strip()]
    if not safe:
        return None
    pat = r"|".join(sorted(safe, key=len, reverse=True))
    return re.compile(rf"\b({pat})\b", re.IGNORECASE)

def make_rule_scorer(include_cues: List[str], exclude_cues: List[str], inc_w=1.0, exc_w=1.0):
    """Create rule-based scorer from cues"""
    inc_pat = compile_cues(include_cues)
    exc_pat = compile_cues(exclude_cues)
    
    def score(txt: str) -> float:
        t = txt if isinstance(txt, str) else str(txt)
        inc = len(inc_pat.findall(t)) if inc_pat else 0
        exc = len(exc_pat.findall(t)) if exc_pat else 0
        return inc_w * inc - exc_w * exc
    
    return score

def precision_recall_f1(y_true, y_score, thresholds=None):
    """Calculate precision, recall, F1 with optimal threshold"""
    if thresholds is None:
        thresholds = np.unique(y_score)
    
    best = {"tau": None, "f1": -1, "precision": 0, "recall": 0}
    y_true = np.array(y_true).astype(int)
    y_score = np.array(y_score, dtype=float)
    
    for t in thresholds:
        yhat = (y_score >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        
        if f1 > best["f1"]:
            best = {
                "tau": float(t), "f1": float(f1), "precision": float(prec), "recall": float(rec),
                "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
            }
    
    # PR curve points
    precisions, recalls, thresh = precision_recall_curve(y_true, y_score)
    pr_points = [{
        "t": float(thresh[i]) if i < len(thresh) else None,
        "precision": float(precisions[i]), 
        "recall": float(recalls[i])
    } for i in range(len(precisions))]
    
    return best, pr_points

def selectivity(hard_neg_scores, tau: float) -> float:
    """Calculate selectivity on hard negatives"""
    if len(hard_neg_scores) == 0:
        return 1.0
    fp = int((np.array(hard_neg_scores) >= tau).sum())
    return float(1.0 - fp / len(hard_neg_scores))

def choose_tau_with_selectivity(y_true, y_score, hard_neg_scores, min_selectivity=0.8):
    """Choose threshold with selectivity constraint"""
    thresholds = np.unique(y_score)
    y_true = np.array(y_true).astype(int)
    best = {"tau": None, "f1": -1}
    cm = None
    
    for t in thresholds:
        yhat = (y_score >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        sel = selectivity(hard_neg_scores, t)
        
        if sel >= min_selectivity and f1 > best["f1"]:
            best = {
                "tau": float(t), "f1": float(f1), "precision": float(prec), 
                "recall": float(rec), "selectivity": float(sel),
                "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
            }
            cm = (tn, fp, fn, tp)
    
    return best if best["tau"] is not None else None

def train_logistic_scorer(positives: List[str], negatives: List[str], 
                         include_cues: List[str], exclude_cues: List[str]):
    """Train logistic regression scorer on cues"""
    vocab = list({*include_cues, *exclude_cues})
    vect = CountVectorizer(vocabulary=vocab, lowercase=True, token_pattern=r"(?u)\b\w+\b")
    X = vect.transform(positives + negatives)
    y = np.array([1] * len(positives) + [0] * len(negatives))
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    
    def score(txt: str):
        return float(clf.predict_proba(vect.transform([txt]))[0, 1])
    
    return score

# ------------------------------
# Robustness testing (fuzzing)
# ------------------------------

def perturbations(txt: str) -> List[str]:
    """Generate text perturbations for robustness testing"""
    outs = []
    outs.append(re.sub(r"\b(increase|rise|up)\b", "growth", txt, flags=re.I))
    outs.append(re.sub(r"\b(decrease|fall|down)\b", "decline", txt, flags=re.I))
    outs.append(re.sub(r"\b(Company|Firm)\b", "Issuer", txt, flags=re.I))
    outs.append(re.sub(r"\b(not|no)\b", "", txt, flags=re.I))
    outs.append(re.sub(r"\d+", "N", txt))
    return list({o for o in outs if o and fuzz.ratio(o, txt) > 0})

def fuzzing_delta(score_fn, positives: List[str], negatives: List[str]) -> Dict[str, float]:
    """Calculate fuzzing robustness deltas"""
    def avg(scores):
        return float(np.mean(scores)) if scores else 0.0
    
    pos_base = [score_fn(x) for x in positives]
    pos_fuzz = [score_fn(y) for x in positives for y in perturbations(x)]
    neg_base = [score_fn(x) for x in negatives]
    neg_fuzz = [score_fn(y) for x in negatives for y in perturbations(x)]
    
    return {
        "pos_mean": avg(pos_base),
        "pos_fuzz_mean": avg(pos_fuzz),
        "neg_mean": avg(neg_base),
        "neg_fuzz_mean": avg(neg_fuzz),
        "pos_drop": avg(pos_base) - avg(pos_fuzz),
        "neg_rise": avg(neg_fuzz) - avg(neg_base)
    }

# ------------------------------
# Main Feature Card builder
# ------------------------------

def build_feature_card(
    base_model_id: str,
    sae_meta: Dict[str, Any],
    latent_id: int,
    domain_name: str,
    positives_train: List[str],
    negatives_train: List[str],
    positives_val: List[str],
    negatives_val: List[str],
    bg_texts_for_hard_negs: List[str],
    min_selectivity_gate: float = 0.8,
    min_f1_gate: float = 0.70,
    outdir: str = "feature_cards",
    use_logistic: bool = False,
    llm_explainer: Optional[MetaLLMExplainer] = None
) -> Tuple[Dict[str, Any], str]:
    """
    Build complete Feature Card for a latent feature
    
    Args:
        base_model_id: Base model identifier
        sae_meta: SAE metadata
        latent_id: Latent feature ID
        domain_name: Domain name
        positives_train: Training positive examples
        negatives_train: Training negative examples
        positives_val: Validation positive examples
        negatives_val: Validation negative examples
        bg_texts_for_hard_negs: Background corpus for hard negatives
        min_selectivity_gate: Minimum selectivity threshold
        min_f1_gate: Minimum F1 threshold
        outdir: Output directory
        use_logistic: Whether to use logistic regression scorer
        llm_explainer: Optional LLM explainer instance
    
    Returns:
        Tuple of (feature_card_dict, output_path)
    """
    
    # Hard negatives via FAISS
    ix = EmbIndex()
    ix.build(bg_texts_for_hard_negs)
    hard_negs = list(itertools.chain.from_iterable(ix.knn(positives_train, k=3)))
    
    # Contrastive explanation
    expl = explain_contrastive(positives_train, hard_negs, n_trials=3, llm_explainer=llm_explainer)
    label = expl["label"]
    
    # Clustering/polysemanticity
    cluster = cluster_exemplars(positives_train + positives_val)
    poly_idx = polysemanticity_index(cluster)
    
    # Scorer(s)
    rule_score = make_rule_scorer(expl["include_cues"], expl["exclude_cues"])
    
    if use_logistic:
        logi_score = train_logistic_scorer(positives_train, negatives_train,
                                         expl["include_cues"], expl["exclude_cues"])
        def blend_score(t):
            return 0.6 * rule_score(t) + 0.4 * logi_score(t)
        score_fn = blend_score
    else:
        score_fn = rule_score
    
    # Validation scores
    y_val = np.array([1] * len(positives_val) + [0] * len(negatives_val))
    X_val = positives_val + negatives_val
    y_score = np.array([score_fn(x) for x in X_val])
    hard_neg_scores = np.array([score_fn(x) for x in hard_negs])
    
    # Threshold with selectivity gate
    best = choose_tau_with_selectivity(y_val, y_score, hard_neg_scores, min_selectivity=min_selectivity_gate)
    
    if best is None:
        best, pr_pts = precision_recall_f1(y_val, y_score)
        select = selectivity(hard_neg_scores, best["tau"])
        best.update({"selectivity": select})
    else:
        _, pr_pts = precision_recall_f1(y_val, y_score)
    
    # Decision logic
    decision = "accept" if (best["f1"] >= min_f1_gate and best["selectivity"] >= min_selectivity_gate) else "revise_or_split"
    if poly_idx >= 0.6:
        decision = "split"
    elif 0.2 <= poly_idx < 0.6 and decision == "accept":
        decision = "accept_but_mark_polysemantic"
    
    # Robustness testing
    fuzz = fuzzing_delta(score_fn, positives_val, negatives_val)
    
    # Build Feature Card
    card = {
        "meta": {
            "base_model": base_model_id,
            "sae": sae_meta,
            "domain": domain_name,
            "latent_id": int(latent_id),
        },
        "labeling": {
            "label": label,
            "definition": expl["definition"],
            "include_cues": expl["include_cues"],
            "exclude_cues": expl["exclude_cues"],
            "risks": expl["risks"],
            "self_consistency": expl["self_consistency"],
        },
        "clustering": {
            "n_clusters": cluster["n_clusters"],
            "silhouette": cluster["silhouette"],
            "polysemanticity_index": poly_idx
        },
        "metrics": {
            "tau_star": best["tau"],
            "f1": best["f1"],
            "precision": best["precision"],
            "recall": best["recall"],
            "selectivity": best["selectivity"],
            "confusion_matrix": {"tn": best["tn"], "fp": best["fp"], "fn": best["fn"], "tp": best["tp"]},
            "pr_curve": pr_pts,
            "robustness_fuzzing": fuzz,
        },
        "canonicals": {
            "positives": positives_train[:10],
            "negatives": negatives_train[:10],
            "hard_negatives": hard_negs[:10]
        },
        "lineage": {
            "data_hash": sha("".join(positives_train + negatives_train + positives_val + negatives_val)),
            "bg_hash": sha("".join(bg_texts_for_hard_negs))
        },
        "decision": decision
    }
    
    # Persist
    out_path = os.path.join(outdir, f"{domain_name}_latent{latent_id}_{label or 'unlabeled'}_{sha(json.dumps(card['metrics']))}.json")
    write_json(card, out_path)
    
    return card, out_path

# ------------------------------
# Batch processing utilities
# ------------------------------

def process_latent_batch(
    latent_data: Dict[int, Dict[str, Any]],
    base_model_id: str,
    sae_meta: Dict[str, Any],
    domain_name: str,
    bg_corpus: List[str],
    outdir: str = "feature_cards",
    llm_explainer: Optional[MetaLLMExplainer] = None,
    **kwargs
) -> List[Tuple[Dict[str, Any], str]]:
    """
    Process a batch of latent features
    
    Args:
        latent_data: Dict mapping latent_id to data
        base_model_id: Base model identifier
        sae_meta: SAE metadata
        domain_name: Domain name
        bg_corpus: Background corpus
        outdir: Output directory
        llm_explainer: Optional LLM explainer
        **kwargs: Additional arguments for build_feature_card
    
    Returns:
        List of (feature_card, output_path) tuples
    """
    results = []
    
    for latent_id, data in latent_data.items():
        try:
            card, path = build_feature_card(
                base_model_id=base_model_id,
                sae_meta=sae_meta,
                latent_id=latent_id,
                domain_name=domain_name,
                positives_train=data["positives_train"],
                negatives_train=data["negatives_train"],
                positives_val=data["positives_val"],
                negatives_val=data["negatives_val"],
                bg_texts_for_hard_negs=bg_corpus,
                outdir=outdir,
                llm_explainer=llm_explainer,
                **kwargs
            )
            results.append((card, path))
            print(f"✅ Processed latent {latent_id}: {card['labeling']['label']}")
            
        except Exception as e:
            print(f"❌ Failed to process latent {latent_id}: {str(e)}")
            continue
    
    return results

# ------------------------------
# Example usage
# ------------------------------

def example_usage():
    """Example usage of the AutoInterp++ pipeline"""
    
    # Sample data (replace with your actual data)
    sample_latent_data = {
        127: {
            "positives_train": [
                "The company reported strong quarterly earnings growth",
                "Revenue increased by 15% year-over-year",
                "Profit margins expanded significantly",
                "Financial performance exceeded expectations"
            ],
            "negatives_train": [
                "The stock price declined today",
                "Market volatility increased",
                "Trading volume was low",
                "Technical indicators show weakness"
            ],
            "positives_val": [
                "Earnings per share beat estimates",
                "Revenue growth accelerated",
                "Operating income improved"
            ],
            "negatives_val": [
                "Share price fell sharply",
                "Market sentiment turned negative",
                "Volume dried up"
            ]
        }
    }
    
    # Background corpus for hard negatives
    bg_corpus = [
        "The weather is nice today",
        "I went to the store",
        "The movie was entertaining",
        "Technology advances rapidly",
        "Education is important",
        "Food tastes delicious",
        "Music is relaxing",
        "Sports are competitive"
    ]
    
    # Initialize LLM explainer (optional)
    llm_explainer = None
    if HAS_TRANSFORMERS:
        try:
            llm_explainer = MetaLLMExplainer("meta-llama/Llama-2-7b-hf")
        except:
            print("Warning: Could not load Meta model, using fallback")
    
    # Process the batch
    results = process_latent_batch(
        latent_data=sample_latent_data,
        base_model_id="llama-2-7b",
        sae_meta={"layer": 4, "l0": 121, "sha": "abc123"},
        domain_name="finance",
        bg_corpus=bg_corpus,
        outdir="feature_cards",
        llm_explainer=llm_explainer
    )
    
    print(f"\nProcessed {len(results)} features")
    for card, path in results:
        print(f"Feature: {card['labeling']['label']}")
        print(f"F1: {card['metrics']['f1']:.3f}")
        print(f"Decision: {card['decision']}")
        print(f"Saved to: {path}\n")

if __name__ == "__main__":
    example_usage()
