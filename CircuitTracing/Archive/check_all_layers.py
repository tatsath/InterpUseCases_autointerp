#!/usr/bin/env python3
"""
Check if the always-on features problem exists across all layers.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open
import os
from collections import Counter

def check_all_layers():
    """Check if always-on features exist across all layers."""
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model_path = "cxllin/Llama2-7b-Finance"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Test texts
    test_texts = [
        "The stock market is performing well today with strong gains.",
        "The weather forecast predicts sunny skies and warm temperatures.",
        "Scientists discovered a new species of butterfly in the Amazon rainforest."
    ]
    
    # Test all layers
    layers_to_test = [4, 10, 16, 22, 28]
    
    print(f"\nTesting {len(test_texts)} texts across {len(layers_to_test)} layers...")
    
    for layer_idx in layers_to_test:
        print(f"\n{'='*60}")
        print(f"LAYER {layer_idx}")
        print(f"{'='*60}")
        
        # Load SAE weights for this layer
        sae_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_finance_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
        layer_path = os.path.join(sae_path, f"layers.{layer_idx}")
        sae_file = os.path.join(layer_path, "sae.safetensors")
        
        try:
            with safe_open(sae_file, framework="pt", device="cpu") as f:
                encoder = f.get_tensor("encoder.weight")
                encoder_bias = f.get_tensor("encoder.bias")
        except Exception as e:
            print(f"Failed to load SAE weights for layer {layer_idx}: {e}")
            continue
        
        feature_frequencies = Counter()
        
        for i, text in enumerate(test_texts):
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[layer_idx + 1]  # +1 because layer 0 is embeddings
            
            # Move encoder to same device and dtype
            encoder = encoder.to(hidden_states.device).to(hidden_states.dtype)
            encoder_bias = encoder_bias.to(hidden_states.device).to(hidden_states.dtype)
            
            # Compute activations
            feature_activations = torch.matmul(hidden_states, encoder.T) + encoder_bias
            
            # Exclude BOS token
            content_activations = feature_activations[:, 1:, :]
            
            # Get max activations
            max_activations = content_activations.max(dim=1)[0].squeeze(0)
            
            # Get top 10 features
            top10_indices = torch.topk(max_activations, 10).indices.cpu().numpy()
            
            # Track features
            for feat in top10_indices:
                feature_frequencies[feat] += 1
        
        # Analyze this layer
        print(f"Features appearing in top 10 across texts:")
        for feat, count in feature_frequencies.most_common(10):
            percentage = (count / len(test_texts)) * 100
            print(f"  Feature {feat:3d}: {count:2d}/{len(test_texts)} texts ({percentage:5.1f}%)")
        
        # Check for always-present features
        always_present = [feat for feat, count in feature_frequencies.items() if count == len(test_texts)]
        print(f"\nAlways present features: {len(always_present)}")
        
        if len(always_present) > 0:
            print(f"  -> PROBLEM: {len(always_present)} features appear in ALL texts")
            print(f"  -> These features: {always_present}")
        else:
            print(f"  -> GOOD: No features appear in ALL texts")
        
        # Check diversity
        unique_features = len(feature_frequencies)
        total_possible = len(test_texts) * 10
        diversity_ratio = unique_features / total_possible
        
        print(f"\nDiversity analysis:")
        print(f"  Unique features: {unique_features}")
        print(f"  Total possible: {total_possible}")
        print(f"  Diversity ratio: {diversity_ratio:.3f}")
        
        if diversity_ratio < 0.3:
            print(f"  -> POOR: Low diversity (ratio < 0.3)")
        elif diversity_ratio < 0.6:
            print(f"  -> MODERATE: Some diversity (ratio < 0.6)")
        else:
            print(f"  -> GOOD: High diversity (ratio >= 0.6)")

if __name__ == "__main__":
    check_all_layers()
