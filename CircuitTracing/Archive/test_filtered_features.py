#!/usr/bin/env python3
"""
Test if filtering out always-on features improves diversity.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open
import os
from collections import Counter

def test_filtered_features():
    """Test if filtering always-on features improves diversity."""
    
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
    
    # Test texts
    test_texts = [
        "The stock market is performing well today with strong gains.",
        "The weather forecast predicts sunny skies and warm temperatures.",
        "Scientists discovered a new species of butterfly in the Amazon rainforest.",
        "The company reported strong quarterly earnings with revenue growth of 15% year-over-year.",
        "Students are preparing for their final exams next week.",
        "The chef prepared a delicious three-course meal for the dinner party."
    ]
    
    # Define always-on features to filter out
    always_on_features = {205, 364, 248, 37, 254, 93, 39, 219, 363, 92, 192, 317}
    
    print(f"\nTesting {len(test_texts)} texts with and without filtering...")
    
    # Test without filtering
    print(f"\n{'='*60}")
    print("WITHOUT FILTERING (original)")
    print(f"{'='*60}")
    
    feature_frequencies_no_filter = Counter()
    
    for i, text in enumerate(test_texts):
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
        
        # Track features
        for feat in top10_indices:
            feature_frequencies_no_filter[feat] += 1
    
    print(f"Features appearing in top 10 across texts:")
    for feat, count in feature_frequencies_no_filter.most_common(10):
        percentage = (count / len(test_texts)) * 100
        print(f"  Feature {feat:3d}: {count:2d}/{len(test_texts)} texts ({percentage:5.1f}%)")
    
    # Test with filtering
    print(f"\n{'='*60}")
    print("WITH FILTERING (excluding always-on features)")
    print(f"{'='*60}")
    
    feature_frequencies_filtered = Counter()
    
    for i, text in enumerate(test_texts):
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
        
        # Filter out always-on features
        filtered_indices = [feat for feat in top10_indices if feat not in always_on_features]
        
        # Track features
        for feat in filtered_indices:
            feature_frequencies_filtered[feat] += 1
    
    print(f"Features appearing in top 10 across texts (after filtering):")
    for feat, count in feature_frequencies_filtered.most_common(10):
        percentage = (count / len(test_texts)) * 100
        print(f"  Feature {feat:3d}: {count:2d}/{len(test_texts)} texts ({percentage:5.1f}%)")
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    
    unique_no_filter = len(feature_frequencies_no_filter)
    unique_filtered = len(feature_frequencies_filtered)
    total_possible = len(test_texts) * 10
    
    diversity_no_filter = unique_no_filter / total_possible
    diversity_filtered = unique_filtered / total_possible
    
    print(f"Without filtering:")
    print(f"  Unique features: {unique_no_filter}")
    print(f"  Diversity ratio: {diversity_no_filter:.3f}")
    
    print(f"With filtering:")
    print(f"  Unique features: {unique_filtered}")
    print(f"  Diversity ratio: {diversity_filtered:.3f}")
    
    print(f"Improvement:")
    print(f"  More unique features: {unique_filtered - unique_no_filter}")
    print(f"  Diversity improvement: {diversity_filtered - diversity_no_filter:.3f}")
    
    if diversity_filtered > diversity_no_filter:
        print(f"  -> SUCCESS: Filtering improved diversity!")
    else:
        print(f"  -> NO IMPROVEMENT: Filtering did not improve diversity")

if __name__ == "__main__":
    test_filtered_features()
