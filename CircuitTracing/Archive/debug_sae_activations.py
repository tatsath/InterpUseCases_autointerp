#!/usr/bin/env python3
"""
Debug script to check SAE activations in detail.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open
import os

def debug_sae_activations():
    """Debug SAE activations in detail."""
    
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
    
    print(f"SAE Encoder shape: {encoder.shape}")
    print(f"SAE Encoder bias shape: {encoder_bias.shape}")
    
    # Test texts
    test_texts = [
        "The stock market is performing well today with strong gains",
        "The weather forecast predicts sunny skies and warm temperatures"
    ]
    
    print(f"\nTesting {len(test_texts)} different texts...")
    
    for i, text in enumerate(test_texts):
        print(f"\n--- Text {i+1}: {text[:50]}... ---")
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[29]  # Layer 28 (0-indexed)
        
        print(f"Hidden states shape: {hidden_states.shape}")
        
        # Move encoder to same device and dtype
        encoder = encoder.to(hidden_states.device).to(hidden_states.dtype)
        encoder_bias = encoder_bias.to(hidden_states.device).to(hidden_states.dtype)
        
        # Compute activations
        feature_activations = torch.matmul(hidden_states, encoder.T) + encoder_bias
        
        print(f"Feature activations shape: {feature_activations.shape}")
        print(f"Feature activations mean: {feature_activations.mean().item():.6f}")
        print(f"Feature activations std: {feature_activations.std().item():.6f}")
        print(f"Feature activations min: {feature_activations.min().item():.6f}")
        print(f"Feature activations max: {feature_activations.max().item():.6f}")
        
        # Check if all activations are the same across sequence length
        print(f"Are all activations identical across sequence? {torch.all(feature_activations[0, 0, :] == feature_activations[0, 1, :]).item()}")
        
        # Check first few positions
        print(f"First 3 positions, first 5 features:")
        for pos in range(min(3, feature_activations.shape[1])):
            print(f"  Position {pos}: {feature_activations[0, pos, :5].cpu().numpy()}")
        
        # Get max activations
        max_activations = feature_activations.max(dim=1)[0].squeeze(0)
        print(f"Max activations shape: {max_activations.shape}")
        print(f"Max activations mean: {max_activations.mean().item():.6f}")
        print(f"Max activations std: {max_activations.std().item():.6f}")
        print(f"Max activations min: {max_activations.min().item():.6f}")
        print(f"Max activations max: {max_activations.max().item():.6f}")
        
        # Get top 10 features
        top10_indices = torch.topk(max_activations, 10).indices
        top10_values = torch.topk(max_activations, 10).values
        
        print(f"Top 10 feature indices: {top10_indices.cpu().numpy()}")
        print(f"Top 10 feature values: {top10_values.cpu().numpy()}")
        
        # Check if the same features are always at the top
        if i == 0:
            first_text_top10 = top10_indices.cpu().numpy()
        else:
            print(f"Overlap with first text: {len(set(first_text_top10) & set(top10_indices.cpu().numpy()))}/10")
            
            # Check if values are identical
            if i == 1:
                first_text_values = torch.topk(max_activations, 10).values.cpu().numpy()
                print(f"Are top 10 values identical? {np.allclose(first_text_values, top10_values.cpu().numpy())}")
                print(f"Max difference in top 10 values: {np.max(np.abs(first_text_values - top10_values.cpu().numpy()))}")

if __name__ == "__main__":
    debug_sae_activations()
