#!/usr/bin/env python3
"""
Debug script to check if the issue exists across multiple layers.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open
import os

def debug_multiple_layers():
    """Check if the issue exists across multiple layers."""
    
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
        "The stock market is performing well today with strong gains",
        "The weather forecast predicts sunny skies and warm temperatures"
    ]
    
    # Test multiple layers
    layers_to_test = [4, 10, 16, 22, 28]
    
    print(f"\nTesting {len(test_texts)} different texts across {len(layers_to_test)} layers...")
    
    for layer_idx in layers_to_test:
        print(f"\n{'='*50}")
        print(f"LAYER {layer_idx}")
        print(f"{'='*50}")
        
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
        
        print(f"SAE Encoder shape: {encoder.shape}")
        
        layer_results = {}
        
        for i, text in enumerate(test_texts):
            print(f"\n--- Text {i+1}: {text[:50]}... ---")
            
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
            
            # Get max activations
            max_activations = feature_activations.max(dim=1)[0].squeeze(0)
            
            # Get top 10 features
            top10_indices = torch.topk(max_activations, 10).indices
            top10_values = torch.topk(max_activations, 10).values
            
            print(f"Top 10 feature indices: {top10_indices.cpu().numpy()}")
            print(f"Top 10 feature values: {top10_values.cpu().numpy()}")
            
            # Store for comparison
            layer_results[text] = {
                'indices': top10_indices.cpu().numpy(),
                'values': top10_values.cpu().numpy()
            }
        
        # Compare between texts for this layer
        if len(layer_results) == 2:
            texts = list(layer_results.keys())
            text1, text2 = texts[0], texts[1]
            
            indices1 = layer_results[text1]['indices']
            indices2 = layer_results[text2]['indices']
            values1 = layer_results[text1]['values']
            values2 = layer_results[text2]['values']
            
            overlap = len(set(indices1) & set(indices2))
            values_identical = np.allclose(values1, values2)
            max_diff = np.max(np.abs(values1 - values2))
            
            print(f"\nLayer {layer_idx} Comparison:")
            print(f"  Top 10 overlap: {overlap}/10")
            print(f"  Values identical: {values_identical}")
            print(f"  Max difference: {max_diff}")

if __name__ == "__main__":
    debug_multiple_layers()
