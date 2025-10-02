#!/usr/bin/env python3
"""
Analyze which features are consistently high across different texts.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open
import os
from collections import Counter

def analyze_consistent_features():
    """Analyze which features are consistently high across different texts."""
    
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
    
    # Load SAE weights for layer 28
    sae_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_finance_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
    layer_path = os.path.join(sae_path, "layers.28")
    sae_file = os.path.join(layer_path, "sae.safetensors")
    
    with safe_open(sae_file, framework="pt", device="cpu") as f:
        encoder = f.get_tensor("encoder.weight")
        encoder_bias = f.get_tensor("encoder.bias")
    
    # Test texts with different content types
    test_texts = [
        "The stock market is performing well today with strong gains.",
        "The weather forecast predicts sunny skies and warm temperatures.",
        "Scientists discovered a new species of butterfly in the Amazon rainforest.",
        "The company reported strong quarterly earnings with revenue growth of 15% year-over-year.",
        "Students are preparing for their final exams next week.",
        "The chef prepared a delicious three-course meal for the dinner party.",
        "The novel explores themes of love, loss, and redemption in modern society.",
        "The football team won the championship with an incredible last-minute goal.",
        "The artist painted a beautiful landscape using vibrant colors and bold strokes.",
        "The doctor recommended regular exercise and a balanced diet for better health."
    ]
    
    print(f"\nAnalyzing feature consistency across {len(test_texts)} diverse texts...")
    
    all_top_features = []
    feature_frequencies = Counter()
    
    for i, text in enumerate(test_texts):
        print(f"\n--- Text {i+1}: {text[:50]}... ---")
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[29]  # Layer 28 (0-indexed)
        
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
        top10_values = torch.topk(max_activations, 10).values.cpu().numpy()
        
        print(f"Top 10 features: {top10_indices}")
        print(f"Top 10 values: {top10_values}")
        
        # Track features
        all_top_features.append(top10_indices)
        for feat in top10_indices:
            feature_frequencies[feat] += 1
    
    # Analyze consistency
    print(f"\n{'='*80}")
    print("FEATURE CONSISTENCY ANALYSIS")
    print(f"{'='*80}")
    
    print(f"\nFeatures appearing in top 10 across texts:")
    for feat, count in feature_frequencies.most_common(20):
        percentage = (count / len(test_texts)) * 100
        print(f"  Feature {feat:3d}: {count:2d}/{len(test_texts)} texts ({percentage:5.1f}%)")
    
    # Check for overly consistent features
    overly_consistent = [feat for feat, count in feature_frequencies.items() if count >= len(test_texts) * 0.8]
    print(f"\nOverly consistent features (appearing in ≥80% of texts): {overly_consistent}")
    
    # Check for features that appear in all texts
    always_present = [feat for feat, count in feature_frequencies.items() if count == len(test_texts)]
    print(f"Always present features (appearing in 100% of texts): {always_present}")
    
    # Analyze feature diversity
    unique_features = len(feature_frequencies)
    total_possible = len(test_texts) * 10
    diversity_ratio = unique_features / total_possible
    
    print(f"\nFeature diversity analysis:")
    print(f"  Total unique features in top 10: {unique_features}")
    print(f"  Total possible features: {total_possible}")
    print(f"  Diversity ratio: {diversity_ratio:.3f}")
    
    # Check if we have good variation
    if diversity_ratio > 0.5:
        print(f"  -> GOOD: High feature diversity across texts")
    elif diversity_ratio > 0.3:
        print(f"  -> MODERATE: Some feature diversity across texts")
    else:
        print(f"  -> POOR: Low feature diversity across texts")
    
    # Analyze the most common features
    print(f"\nMost common features analysis:")
    for feat, count in feature_frequencies.most_common(5):
        percentage = (count / len(test_texts)) * 100
        print(f"  Feature {feat}: appears in {count}/{len(test_texts)} texts ({percentage:.1f}%)")
        
        # Check if this feature appears in all texts
        if count == len(test_texts):
            print(f"    -> WARNING: This feature appears in ALL texts (might be always-on)")

if __name__ == "__main__":
    analyze_consistent_features()
