#!/usr/bin/env python3
"""
Test script to verify that excluding the BOS token fixes the activation variation issue.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open
import os

def test_fixed_activations():
    """Test if excluding BOS token fixes the activation variation issue."""
    
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
    
    # Test texts - more diverse
    test_texts = [
        "The stock market is performing well today with strong gains",
        "The weather forecast predicts sunny skies and warm temperatures", 
        "Scientists discovered a new species of butterfly in the Amazon rainforest",
        "The company reported strong quarterly earnings with revenue growth of 15% year-over-year",
        "The chef prepared a delicious three-course meal for the dinner party",
        "Students are preparing for their final exams next week"
    ]
    
    print(f"\nTesting {len(test_texts)} different texts with BOS token excluded...")
    
    activations_by_text = {}
    
    for i, text in enumerate(test_texts):
        print(f"\n--- Text {i+1}: {text[:50]}... ---")
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        print(f"Input tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")
        print(f"Input shape: {inputs['input_ids'].shape}")
        
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
        
        # Exclude BOS token (first token)
        content_activations = feature_activations[:, 1:, :]
        print(f"Content activations shape (excluding BOS): {content_activations.shape}")
        
        # Use max activations on content only
        max_activations = content_activations.max(dim=1)[0].squeeze(0)
        
        # Get top 10 features
        top10_indices = torch.topk(max_activations, 10).indices
        top10_values = torch.topk(max_activations, 10).values
        
        print(f"Top 10 feature indices: {top10_indices.cpu().numpy()}")
        print(f"Top 10 feature values: {top10_values.cpu().numpy()}")
        
        # Store for comparison
        activations_by_text[text] = {
            'indices': top10_indices.cpu().numpy(),
            'values': top10_values.cpu().numpy()
        }
    
    # Compare activations between texts
    print(f"\n{'='*60}")
    print("COMPARISON BETWEEN TEXTS (BOS TOKEN EXCLUDED)")
    print(f"{'='*60}")
    
    texts = list(activations_by_text.keys())
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            text1, text2 = texts[i], texts[j]
            
            indices1 = activations_by_text[text1]['indices']
            indices2 = activations_by_text[text2]['indices']
            values1 = activations_by_text[text1]['values']
            values2 = activations_by_text[text2]['values']
            
            overlap = len(set(indices1) & set(indices2))
            values_identical = np.allclose(values1, values2)
            max_diff = np.max(np.abs(values1 - values2))
            
            print(f"\nText {i+1} vs Text {j+1}:")
            print(f"  Top 10 overlap: {overlap}/10")
            print(f"  Values identical: {values_identical}")
            print(f"  Max difference: {max_diff:.6f}")
            print(f"  Text {i+1} top 5: {indices1[:5]}")
            print(f"  Text {j+1} top 5: {indices2[:5]}")

if __name__ == "__main__":
    test_fixed_activations()
