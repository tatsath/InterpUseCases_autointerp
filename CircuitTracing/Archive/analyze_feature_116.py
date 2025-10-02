#!/usr/bin/env python3
"""
Analyze where feature 116 is activating across different token positions.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open
import os

def analyze_feature_116():
    """Analyze where feature 116 is activating across token positions."""
    
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
    
    # Test texts with different structures
    test_texts = [
        "The stock market is performing well today with strong gains.",
        "The weather forecast predicts sunny skies and warm temperatures.",
        "Scientists discovered a new species of butterfly in the Amazon rainforest.",
        "The company reported strong quarterly earnings with revenue growth of 15% year-over-year, driven by increased demand in the technology sector.",
        "Students are preparing for their final exams next week.",
        "The chef prepared a delicious three-course meal for the dinner party."
    ]
    
    print(f"\nAnalyzing feature 116 activation patterns across {len(test_texts)} texts...")
    
    for i, text in enumerate(test_texts):
        print(f"\n{'='*80}")
        print(f"TEXT {i+1}: {text}")
        print(f"{'='*80}")
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        print(f"Tokens: {tokens}")
        print(f"Number of tokens: {len(tokens)}")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[29]  # Layer 28 (0-indexed)
        
        # Move encoder to same device and dtype
        encoder = encoder.to(hidden_states.device).to(hidden_states.dtype)
        encoder_bias = encoder_bias.to(hidden_states.device).to(hidden_states.dtype)
        
        # Compute activations
        feature_activations = torch.matmul(hidden_states, encoder.T) + encoder_bias
        
        # Focus on feature 116
        feature_116_activations = feature_activations[0, :, 116].cpu().numpy()
        
        print(f"\nFeature 116 activations across all positions:")
        for pos, (token, activation) in enumerate(zip(tokens, feature_116_activations)):
            print(f"  Position {pos:2d}: {token:15s} -> {activation:8.3f}")
        
        # Find where feature 116 is highest
        max_pos = np.argmax(feature_116_activations)
        print(f"\nFeature 116 highest activation:")
        print(f"  Position: {max_pos}")
        print(f"  Token: {tokens[max_pos]}")
        print(f"  Activation: {feature_116_activations[max_pos]:.3f}")
        
        # Check if it's at the beginning or end
        if max_pos == 0:
            print(f"  -> HIGHEST at BOS token (beginning)")
        elif max_pos == len(tokens) - 1:
            print(f"  -> HIGHEST at last token (end)")
        else:
            print(f"  -> HIGHEST at middle position")
        
        # Check if it's high at sentence boundaries
        sentence_endings = ['.', '!', '?']
        high_activations = feature_116_activations > np.percentile(feature_116_activations, 75)
        
        print(f"\nFeature 116 high activations (>75th percentile):")
        for pos, (token, activation, is_high) in enumerate(zip(tokens, feature_116_activations, high_activations)):
            if is_high:
                boundary_info = ""
                if token in sentence_endings:
                    boundary_info = " [SENTENCE END]"
                elif pos == 0:
                    boundary_info = " [BOS]"
                elif pos == len(tokens) - 1:
                    boundary_info = " [EOS]"
                print(f"  Position {pos:2d}: {token:15s} -> {activation:8.3f}{boundary_info}")
        
        # Compare with other top features
        max_activations = feature_activations.max(dim=1)[0].squeeze(0)
        top5_indices = torch.topk(max_activations, 5).indices.cpu().numpy()
        top5_values = torch.topk(max_activations, 5).values.cpu().numpy()
        
        print(f"\nTop 5 features (excluding BOS):")
        for feat_idx, value in zip(top5_indices, top5_values):
            print(f"  Feature {feat_idx}: {value:.3f}")
        
        # Check if feature 116 is still dominating
        if 116 in top5_indices:
            rank = np.where(top5_indices == 116)[0][0] + 1
            print(f"  -> Feature 116 is #{rank} (still dominating!)")
        else:
            print(f"  -> Feature 116 is NOT in top 5 (good!)")
        
        # Analyze activation pattern for feature 116
        print(f"\nFeature 116 activation pattern analysis:")
        print(f"  Mean activation: {feature_116_activations.mean():.3f}")
        print(f"  Std activation: {feature_116_activations.std():.3f}")
        print(f"  Min activation: {feature_116_activations.min():.3f}")
        print(f"  Max activation: {feature_116_activations.max():.3f}")
        
        # Check if it's consistently high
        high_threshold = feature_116_activations.mean() + feature_116_activations.std()
        consistently_high = np.sum(feature_116_activations > high_threshold)
        print(f"  Positions above mean+std: {consistently_high}/{len(feature_116_activations)}")

if __name__ == "__main__":
    analyze_feature_116()
