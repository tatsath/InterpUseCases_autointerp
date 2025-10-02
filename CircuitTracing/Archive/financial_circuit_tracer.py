"""
Financial Circuit Tracer for SAE Feature Analysis

This module provides comprehensive circuit tracing capabilities for analyzing
feature relationships across multiple SAE layers in financial contexts.

Key Features:
- Multi-layer SAE feature activation analysis
- Attention-mediated feature graph construction
- Financial topic-specific circuit tracing
- Comprehensive visualization and export capabilities
"""

import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open
import warnings
warnings.filterwarnings("ignore")

@dataclass
class SAEWrapper:
    """Wrapper for trained SAE modules per layer."""
    layer: int
    encoder: torch.Tensor
    encoder_bias: Optional[torch.Tensor] = None
    decoder: Optional[torch.Tensor] = None
    decoder_bias: Optional[torch.Tensor] = None
    
    def encode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Encode hidden states to SAE features."""
        # Compute activations: (batch_size, seq_len, n_features)
        feature_activations = torch.matmul(hidden_states, self.encoder.T)
        
        # Add encoder bias if available
        if self.encoder_bias is not None:
            feature_activations = feature_activations + self.encoder_bias
        
        # Apply ReLU activation
        return torch.relu(feature_activations)

class FinancialTopicData:
    """Financial topic datasets for circuit tracing analysis."""
    
    def __init__(self):
        self.topics = {
            "mergers_acquisitions": {
                "keywords": ["merger", "acquisition", "takeover", "deal", "consolidation", "buyout"],
                "prompts": [
                    "Microsoft's acquisition of Activision Blizzard for $68.7 billion represents the largest gaming industry merger in history, reshaping the competitive landscape and regulatory environment.",
                    "The proposed merger between T-Mobile and Sprint faced intense regulatory scrutiny before receiving approval, creating a stronger competitor to Verizon and AT&T.",
                    "Berkshire Hathaway's acquisition strategy focuses on companies with strong competitive moats and predictable cash flows, as demonstrated by recent deals in energy and insurance sectors."
                ]
            },
            "financial_leaders": {
                "keywords": ["ceo", "chairman", "executive", "leadership", "director", "president"],
                "prompts": [
                    "Jamie Dimon's leadership at JPMorgan Chase has been marked by strategic acquisitions, digital transformation initiatives, and navigating multiple financial crises while maintaining strong capital ratios.",
                    "Warren Buffett's investment philosophy and shareholder letters provide insights into value investing, market timing, and long-term wealth creation strategies.",
                    "Elon Musk's unconventional leadership style at Tesla has driven innovation in electric vehicles while creating significant market volatility and regulatory challenges."
                ]
            },
            "financial_entities": {
                "keywords": ["bank", "hedge fund", "private equity", "investment", "corporation", "institution"],
                "prompts": [
                    "Goldman Sachs' investment banking division reported record revenues driven by strong M&A activity and capital markets transactions across technology and healthcare sectors.",
                    "BlackRock's asset management business continues to dominate the passive investing space while expanding into alternative investments and sustainable finance solutions.",
                    "The Federal Reserve's balance sheet expansion through quantitative easing programs has fundamentally altered the global financial system's liquidity dynamics."
                ]
            },
            "financial_regulations": {
                "keywords": ["regulation", "compliance", "sec", "federal reserve", "policy", "oversight"],
                "prompts": [
                    "The SEC's new climate disclosure rules require public companies to report greenhouse gas emissions and climate-related risks, significantly impacting corporate reporting standards.",
                    "Basel III capital requirements have forced banks to maintain higher capital ratios and implement more sophisticated risk management frameworks across all business lines.",
                    "The Federal Reserve's stress testing program evaluates banks' ability to withstand economic downturns, influencing dividend policies and capital allocation decisions."
                ]
            },
            "market_events_sentiments": {
                "keywords": ["volatility", "crash", "rally", "sentiment", "fear", "greed", "recession"],
                "prompts": [
                    "The 2008 financial crisis fundamentally changed banking regulations and investor risk tolerance, leading to a decade of ultra-low interest rates and quantitative easing programs.",
                    "The GameStop short squeeze highlighted the power of retail investors and social media in driving market movements, challenging traditional institutional investment strategies.",
                    "The COVID-19 pandemic created unprecedented market volatility, with the S&P 500 experiencing both the fastest bear market and recovery in history."
                ]
            }
        }
    
    def get_topic_prompts(self, topic: str) -> List[str]:
        """Get prompts for a specific financial topic."""
        return self.topics.get(topic, {}).get("prompts", [])
    
    def get_topic_keywords(self, topic: str) -> List[str]:
        """Get keywords for a specific financial topic."""
        return self.topics.get(topic, {}).get("keywords", [])
    
    def get_all_topics(self) -> List[str]:
        """Get all available financial topics."""
        return list(self.topics.keys())

class ActivationTracer:
    """Main class for tracing feature activations across SAE layers."""
    
    def __init__(self, model, tokenizer, sae_wrappers: List[SAEWrapper], device="cuda"):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.layers = sorted([w.layer for w in sae_wrappers])
        self.sae_by_layer = {w.layer: w for w in sae_wrappers}
        self._hooks = []
        self._cache_h = {}         # hidden states by layer: [B, T, d]
        self._cache_attn = {}      # attention weights by layer: [B, H, T, T]
    
    def _hook_hidden(self, layer_idx):
        """Hook to capture hidden states after each transformer layer."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                hs = output[0]
            else:
                hs = output
            self._cache_h[layer_idx] = hs.detach().to("cpu")
        return hook
    
    def _hook_attn(self, layer_idx):
        """Hook to capture attention weights."""
        def attn_hook(module, input, output):
            # Try to get attention weights from the output
            if isinstance(output, tuple) and len(output) > 1:
                attn_probs = output[1]
                if attn_probs is not None:
                    self._cache_attn[layer_idx] = attn_probs.detach().to("cpu")
        return attn_hook
    
    def _register_hooks(self):
        """Register forward hooks for all SAE layers."""
        for li in self.layers:
            if li < len(self.model.model.layers):
                block = self.model.model.layers[li]
                # Hook for hidden states
                self._hooks.append(block.register_forward_hook(self._hook_hidden(li)))
                # Hook for attention weights
                if hasattr(block, "self_attn"):
                    self._hooks.append(block.self_attn.register_forward_hook(self._hook_attn(li)))
    
    def _clear_cache(self):
        """Clear cached activations and attention weights."""
        self._cache_h.clear()
        self._cache_attn.clear()
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []
    
    @torch.no_grad()
    def collect_activations(self, prompts: List[str], max_new_tokens: int = 0) -> Tuple[Dict, Dict, Dict]:
        """Collect SAE feature activations and attention weights for given prompts."""
        self._clear_cache()
        self._register_hooks()
        
        # Tokenize inputs
        toks = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        # Move to the same device as the model
        toks = {k: v.to(self.device) for k, v in toks.items()}
        
        # Forward pass
        if max_new_tokens > 0:
            _ = self.model.generate(**toks, max_new_tokens=max_new_tokens, output_attentions=True)
        else:
            _ = self.model(**toks, output_attentions=True)
        
        # Remove hooks
        self._remove_hooks()
        
        # Compute SAE feature activations per layer
        feats_by_layer = {}
        with torch.no_grad():
            for li in self.layers:
                if li not in self._cache_h:
                    continue
                hs = self._cache_h[li]  # [B, T, d]
                sae = self.sae_by_layer[li]
                feats = sae.encode(hs.to(self.device)).detach().to("cpu")  # [B, T, F]
                feats_by_layer[li] = feats
        
        # Process attention weights
        attn_by_layer = {}
        for li in self.layers:
            if li in self._cache_attn:
                aw = self._cache_attn[li]
                if aw.dim() == 4:  # [B, H, T, T]
                    attn_by_layer[li] = aw.cpu()
        
        return toks, feats_by_layer, attn_by_layer

def build_feature_graph(
    feats_by_layer: Dict[int, torch.Tensor],
    attn_by_layer: Dict[int, torch.Tensor],
    attention_weight: float = 0.6,
    lag_weight: float = 0.3,
    same_layer_weight: float = 0.1,
    topk_edges_per_node: int = 5,
    token_mask: Optional[torch.Tensor] = None
) -> nx.DiGraph:
    """
    Build a directed graph representing feature relationships across layers.
    
    Args:
        feats_by_layer: Dictionary mapping layer indices to feature activations
        attn_by_layer: Dictionary mapping layer indices to attention weights
        attention_weight: Weight for attention-mediated connections
        lag_weight: Weight for same-token lagged connections
        same_layer_weight: Weight for same-layer correlations
        topk_edges_per_node: Maximum number of edges per node
        token_mask: Mask for valid tokens
    
    Returns:
        NetworkX directed graph with feature relationships
    """
    G = nx.DiGraph()
    
    # Prepare token mask
    if token_mask is not None:
        m = token_mask.cpu().numpy().astype(bool)
    else:
        first_layer = next(iter(feats_by_layer.keys()))
        B, T, _ = feats_by_layer[first_layer].shape
        m = np.ones((B, T), dtype=bool)
    
    # Add nodes for all features across all layers
    for L, feats in feats_by_layer.items():
        _, _, F = feats.shape
        for f in range(F):
            G.add_node((L, f))
    
    # Convert to numpy for easier processing
    feats_np = {L: v.numpy() for L, v in feats_by_layer.items()}
    layers_sorted = sorted(feats_by_layer.keys())
    
    # Build edges between consecutive layers
    for idx, L in enumerate(layers_sorted):
        F_L = feats_np[L].shape[-1]
        
        # Same-layer correlations (optional)
        if same_layer_weight > 0:
            X = feats_np[L]  # [B, T, F]
            X_masked = X[m]  # [N, F] where N is number of valid tokens
            if X_masked.shape[0] > 0:
                X_centered = X_masked - X_masked.mean(axis=0, keepdims=True)
                # Compute correlation matrix (sample subset for efficiency)
                subset_size = min(256, X_centered.shape[0])
                if subset_size > 0:
                    subset_indices = np.random.choice(X_centered.shape[0], subset_size, replace=False)
                    X_subset = X_centered[subset_indices]
                    C = np.corrcoef(X_subset.T)  # [F, F]
                    C = np.nan_to_num(C, nan=0.0)
                    
                    for i in range(F_L):
                        nbrs = np.argsort(-np.abs(C[i]))[1:topk_edges_per_node+1]
                        for j in nbrs:
                            w = same_layer_weight * max(0.0, float(abs(C[i, j])))
                            if w > 0:
                                G.add_edge((L, i), (L, j), weight=w, kind="same_layer_corr")
        
        # Cross-layer edges to next layer
        if idx + 1 < len(layers_sorted):
            Lp1 = layers_sorted[idx + 1]
            X = feats_np[L]     # [B, T, F_L]
            Y = feats_np[Lp1]   # [B, T, F_Lp1]
            F_Lp1 = Y.shape[-1]
            
            # Attention-mediated connections
            if attention_weight > 0 and L in attn_by_layer:
                A = attn_by_layer[L].numpy()  # [B, H, T, T]
                A_mean = A.mean(axis=1)       # [B, T, T] - average across heads
                
                # Get top features per token for efficiency
                top_q = 5
                X_top = np.argsort(-X, axis=-1)[..., :top_q]    # [B, T, top_q]
                Y_top = np.argsort(-Y, axis=-1)[..., :top_q]    # [B, T, top_q]
                
                B, T, _ = X.shape
                for b in range(B):
                    for tgt in range(T):
                        if not m[b, tgt]:
                            continue
                        src_weights = A_mean[b, tgt]  # [T] - attention from tgt to all sources
                        src_indices = np.argsort(-src_weights)[:3]  # Top 3 source tokens
                        
                        for s in src_indices:
                            if not m[b, s]:
                                continue
                            a_w = float(src_weights[s])
                            if a_w <= 0:
                                continue
                            
                            src_feats = X_top[b, s]    # Top features at source
                            tgt_feats = Y_top[b, tgt]  # Top features at target
                            
                            # Connect feature pairs with attention-weighted scores
                            for fi in src_feats:
                                xi = X[b, s, fi]
                                for fj in tgt_feats:
                                    yj = Y[b, tgt, fj]
                                    score = attention_weight * a_w * float(max(0.0, xi * yj))
                                    if score > 0:
                                        u, v = (L, int(fi)), (Lp1, int(fj))
                                        w_prev = G[u][v]["weight"] if G.has_edge(u, v) else 0.0
                                        G.add_edge(u, v, weight=max(w_prev, score), kind="attn_mediated")
            
            # Lagged connections (same-token influence)
            if lag_weight > 0:
                X_masked = X[m].reshape(-1, F_L)      # [N, F_L]
                Y_masked = Y[m].reshape(-1, F_Lp1)    # [N, F_Lp1]
                
                if X_masked.shape[0] > 0:
                    # Sample subset for efficiency
                    i_subset = min(256, F_L)
                    j_subset = min(256, F_Lp1)
                    I = np.random.choice(F_L, i_subset, replace=False)
                    J = np.random.choice(F_Lp1, j_subset, replace=False)
                    
                    Xs = X_masked[:, I]  # [N, i_subset]
                    Ys = Y_masked[:, J]  # [N, j_subset]
                    
                    # Normalize
                    Xs_norm = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-8)
                    Ys_norm = (Ys - Ys.mean(0)) / (Ys.std(0) + 1e-8)
                    
                    # Compute correlations
                    C = Xs_norm.T @ Ys_norm / Xs_norm.shape[0]  # [i_subset, j_subset]
                    C = np.clip(C, 0, None)  # Only positive influences
                    
                    for a, i in enumerate(I):
                        nbrs = np.argsort(-C[a])[:topk_edges_per_node]
                        for b_, j in enumerate(J[nbrs]):
                            w = lag_weight * float(C[a, nbrs[b_]])
                            if w > 0:
                                G.add_edge((L, int(i)), (Lp1, int(j)), weight=w, kind="lagged")
    
    # Normalize edge weights to [0, 1]
    if G.number_of_edges() > 0:
        wmax = max(d["weight"] for _, _, d in G.edges(data=True))
        if wmax > 0:
            for u, v, d in G.edges(data=True):
                d["weight"] = float(d["weight"] / wmax)
    
    return G

def pick_start_end_features(
    feats_by_layer: Dict[int, torch.Tensor],
    start_layer: int,
    end_layer: int,
    tokenizer,
    input_ids: torch.Tensor,
    start_token_keywords: Optional[List[str]] = None,
    topk_start: int = 5,
    topk_end: int = 5
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Select start and end features for circuit tracing.
    
    Args:
        feats_by_layer: Feature activations by layer
        start_layer: Layer to select start features from
        end_layer: Layer to select end features from
        tokenizer: Tokenizer for decoding tokens
        input_ids: Input token IDs
        start_token_keywords: Keywords to focus on for start features
        topk_start: Number of start features to select
        topk_end: Number of end features to select
    
    Returns:
        Tuple of (start_features, end_features) where each is a list of (layer, feature_id) tuples
    """
    B, T = input_ids.shape
    text_toks = [[tokenizer.decode([tid]) for tid in input_ids[b].tolist()] for b in range(B)]
    
    def mask_for_keywords():
        if not start_token_keywords:
            return np.ones((B, T), dtype=bool)
        mk = np.zeros((B, T), dtype=bool)
        kws = [k.lower() for k in start_token_keywords]
        for b in range(B):
            for t in range(T):
                tok = text_toks[b][t].lower()
                if any(kw in tok for kw in kws):
                    mk[b, t] = True
        if not mk.any():
            mk[:] = True
        return mk
    
    m_start = mask_for_keywords()
    
    # Start features
    X = feats_by_layer[start_layer].numpy()  # [B, T, F]
    X_masked = np.where(m_start[..., None], X, 0.0)
    start_scores = X_masked.max(axis=(0, 1))  # [F]
    start_ids = np.argsort(-start_scores)[:topk_start]
    
    # End features (focus on later tokens)
    Y = feats_by_layer[end_layer].numpy()
    last_k = max(1, T // 4)  # Last quarter of tokens
    Y_masked = np.zeros_like(Y)
    Y_masked[:, -last_k:, :] = Y[:, -last_k:, :]
    end_scores = Y_masked.max(axis=(0, 1))
    end_ids = np.argsort(-end_scores)[:topk_end]
    
    return [(start_layer, int(i)) for i in start_ids], [(end_layer, int(j)) for j in end_ids]

def k_best_paths(
    G: nx.DiGraph, 
    sources: List[Tuple[int, int]], 
    targets: List[Tuple[int, int]], 
    k: int = 3, 
    max_hops: int = 8
) -> List[Dict[str, Any]]:
    """
    Find top-k paths maximizing product of edge weights.
    
    Args:
        G: Feature relationship graph
        sources: List of source (layer, feature) tuples
        targets: List of target (layer, feature) tuples
        k: Number of best paths to return
        max_hops: Maximum number of hops in a path
    
    Returns:
        List of dictionaries containing path information
    """
    import heapq
    import math
    
    # Convert to log costs for shortest path algorithm
    H = nx.DiGraph()
    for u, v, d in G.edges(data=True):
        w = max(d["weight"], 1e-12)  # Ensure positive weight
        # Use negative log for cost (lower cost = higher weight)
        cost = -math.log(w)
        # Ensure cost is positive for shortest path algorithm
        cost = max(cost, 1e-6)
        H.add_edge(u, v, cost=cost, weight=d["weight"], kind=d.get("kind", ""))
    
    # Find paths between all source-target pairs
    heap = []
    for s in sources:
        for t in targets:
            try:
                for p in nx.shortest_simple_paths(H, source=s, target=t, weight="cost"):
                    if len(p) - 1 > max_hops:
                        continue
                    
                    # Compute product weight
                    prod = 1.0
                    kinds = []
                    for i in range(len(p) - 1):
                        d = H[p[i]][p[i + 1]]
                        prod *= max(1e-12, d["weight"])
                        kinds.append(d.get("kind", ""))
                    
                    heapq.heappush(heap, (-prod, p, kinds))
                    if len(heap) > k * 10:
                        heapq.heappop(heap)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
    
    # Get best k paths
    top = sorted(heap, key=lambda x: x[0])[:k]
    results = []
    for negw, p, kinds in top:
        results.append({
            "path": p,
            "edge_kinds": kinds,
            "weight_product": float(-negw)
        })
    
    return results

class FinancialCircuitTracer:
    """Main class for financial circuit tracing analysis."""
    
    def __init__(self, model_path: str, sae_path: str, device: str = "cuda"):
        self.device = device
        self.model_path = model_path
        self.sae_path = sae_path
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Use single GPU for stability
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=None,  # Disable automatic device mapping
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Move model to single GPU
        self.model = self.model.to(self.device)
        
        # Get the primary device (first GPU or CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        
        # Load SAE weights
        self.sae_wrappers = self._load_sae_weights()
        
        # Initialize activation tracer
        self.tracer = ActivationTracer(self.model, self.tokenizer, self.sae_wrappers, device)
        
        # Initialize financial topic data
        self.topic_data = FinancialTopicData()
    
    def _load_sae_weights(self) -> List[SAEWrapper]:
        """Load SAE weights for all layers."""
        sae_wrappers = []
        layers = [4, 10, 16, 22, 28]  # SAE layers
        
        for layer in layers:
            layer_path = os.path.join(self.sae_path, f"layers.{layer}")
            sae_file = os.path.join(layer_path, "sae.safetensors")
            
            if os.path.exists(sae_file):
                with safe_open(sae_file, framework="pt", device="cpu") as f:
                    encoder = f.get_tensor("encoder.weight")
                    encoder_bias = f.get_tensor("encoder.bias")
                    decoder = f.get_tensor("W_dec")
                    decoder_bias = f.get_tensor("b_dec")
                
                # Convert to the same dtype as the model and move to device
                model_dtype = next(self.model.parameters()).dtype
                sae_wrappers.append(SAEWrapper(
                    layer=layer,
                    encoder=encoder.to(self.device, dtype=model_dtype),
                    encoder_bias=encoder_bias.to(self.device, dtype=model_dtype) if encoder_bias is not None else None,
                    decoder=decoder.to(self.device, dtype=model_dtype) if decoder is not None else None,
                    decoder_bias=decoder_bias.to(self.device, dtype=model_dtype) if decoder_bias is not None else None
                ))
                print(f"Loaded SAE weights for layer {layer}")
            else:
                print(f"Warning: SAE weights not found for layer {layer}")
        
        return sae_wrappers
    
    def trace_circuits_for_prompt(
        self,
        prompt: str,
        topic: str = "general",
        start_layer: int = 4,
        end_layer: int = 28,
        k_paths: int = 5,
        attention_weight: float = 0.6,
        lag_weight: float = 0.3,
        same_layer_weight: float = 0.1
    ) -> Tuple[List[Dict], nx.DiGraph, Dict]:
        """
        Trace circuits for a given prompt.
        
        Args:
            prompt: Input prompt to analyze
            topic: Financial topic category
            start_layer: Starting layer for circuit tracing
            end_layer: Ending layer for circuit tracing
            k_paths: Number of top paths to return
            attention_weight: Weight for attention-mediated connections
            lag_weight: Weight for lagged connections
            same_layer_weight: Weight for same-layer correlations
        
        Returns:
            Tuple of (circuit_paths, feature_graph, activations)
        """
        # Get topic-specific keywords
        keywords = self.topic_data.get_topic_keywords(topic)
        
        # Collect activations
        toks, feats_by_layer, attn_by_layer = self.tracer.collect_activations([prompt])
        input_ids = toks["input_ids"].cpu()
        
        # Create token mask
        mask = (input_ids != self.tokenizer.pad_token_id) if self.tokenizer.pad_token_id is not None else torch.ones_like(input_ids, dtype=torch.bool)
        
        # Build feature graph
        G = build_feature_graph(
            feats_by_layer, attn_by_layer,
            attention_weight=attention_weight,
            lag_weight=lag_weight,
            same_layer_weight=same_layer_weight,
            topk_edges_per_node=5,
            token_mask=mask
        )
        
        # Select start and end features
        sources, targets = pick_start_end_features(
            feats_by_layer, start_layer, end_layer,
            self.tokenizer, input_ids,
            start_token_keywords=keywords,
            topk_start=5, topk_end=5
        )
        
        # Find best paths
        results = k_best_paths(G, sources, targets, k=k_paths, max_hops=len(self.tracer.layers))
        
        return results, G, {
            "feats_by_layer": feats_by_layer,
            "attn_by_layer": attn_by_layer,
            "sources": sources,
            "targets": targets
        }
    
    def analyze_topic(self, topic: str, **kwargs) -> Dict[str, Any]:
        """Analyze all prompts for a given financial topic."""
        prompts = self.topic_data.get_topic_prompts(topic)
        keywords = self.topic_data.get_topic_keywords(topic)
        
        results = {}
        for i, prompt in enumerate(prompts):
            print(f"\n=== Analyzing {topic} prompt {i+1}/{len(prompts)} ===")
            print(f"Prompt: {prompt[:100]}...")
            
            try:
                circuit_paths, graph, activations = self.trace_circuits_for_prompt(
                    prompt, topic=topic, **kwargs
                )
                
                results[f"prompt_{i+1}"] = {
                    "prompt": prompt,
                    "circuit_paths": circuit_paths,
                    "graph_stats": {
                        "nodes": graph.number_of_nodes(),
                        "edges": graph.number_of_edges()
                    },
                    "activations": activations
                }
                
                # Print circuit paths
                print(f"\nTop {len(circuit_paths)} circuit paths:")
                for j, result in enumerate(circuit_paths):
                    hops = " -> ".join([f"L{L}:F{f}" for (L, f) in result["path"]])
                    print(f"  {j+1}. [weight={result['weight_product']:.4f}] {hops}")
                    print(f"     Edge types: {result['edge_kinds']}")
                
            except Exception as e:
                print(f"Error analyzing prompt {i+1}: {str(e)}")
                results[f"prompt_{i+1}"] = {"error": str(e)}
        
        return results
    
    def export_results(self, results: Dict[str, Any], output_dir: str = "circuit_tracing_results"):
        """Export analysis results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Export circuit paths as JSON
        with open(os.path.join(output_dir, "circuit_paths.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        # Export feature graphs
        for topic, topic_results in results.items():
            if isinstance(topic_results, dict) and "circuit_paths" in topic_results:
                graph = topic_results.get("graph")
                if graph is not None:
                    # Export graph as edge list
                    edges = []
                    for u, v, d in graph.edges(data=True):
                        edges.append({
                            "source": {"layer": u[0], "feature": u[1]},
                            "target": {"layer": v[0], "feature": v[1]},
                            "weight": d["weight"],
                            "kind": d.get("kind", "")
                        })
                    
                    with open(os.path.join(output_dir, f"{topic}_graph.json"), "w") as f:
                        json.dump({"edges": edges}, f, indent=2)
        
        print(f"Results exported to {output_dir}/")

def main():
    """Main function to run financial circuit tracing analysis."""
    # Configuration
    model_path = "meta-llama/Llama-2-7b-hf"
    sae_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Initializing Financial Circuit Tracer...")
    tracer = FinancialCircuitTracer(model_path, sae_path, device)
    
    # Analyze all financial topics
    all_topics = tracer.topic_data.get_all_topics()
    print(f"Available topics: {all_topics}")
    
    all_results = {}
    for topic in all_topics:
        print(f"\n{'='*60}")
        print(f"ANALYZING TOPIC: {topic.upper()}")
        print(f"{'='*60}")
        
        topic_results = tracer.analyze_topic(topic)
        all_results[topic] = topic_results
    
    # Export all results
    tracer.export_results(all_results)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print("Results have been exported to circuit_tracing_results/")

if __name__ == "__main__":
    main()
