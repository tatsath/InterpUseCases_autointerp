#!/usr/bin/env python3
"""
Test script to verify that feature activations change between different text inputs.
This will help identify why the rankings aren't changing much.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open
import os

def test_activation_variation():
    """Test if activations actually change between different texts."""
    
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
        "The stock market is performing well today with strong gains",
        "The weather forecast predicts sunny skies and warm temperatures", 
        "Scientists discovered a new species of butterfly in the Amazon rainforest",
        "The company reported strong quarterly earnings with revenue growth of 15% year-over-year"
    ]
    
    print(f"\nTesting {len(test_texts)} different texts...")
    
    activations_by_text = {}
    
    for i, text in enumerate(test_texts):
        print(f"\n--- Text {i+1}: {text[:50]}... ---")
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        print(f"Input shape: {inputs['input_ids'].shape}")
        print(f"Token length: {inputs['input_ids'].shape[1]}")
        
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
        
        # Test different aggregation methods
        mean_activations = feature_activations.mean(dim=1).squeeze(0)
        max_activations = feature_activations.max(dim=1)[0].squeeze(0)
        sum_activations = feature_activations.sum(dim=1).squeeze(0)
        
        # Get top 10 features for each method
        mean_top10 = torch.topk(mean_activations, 10)
        max_top10 = torch.topk(max_activations, 10)
        sum_top10 = torch.topk(sum_activations, 10)
        
        print(f"Mean activations - Top 10 features: {mean_top10.indices.cpu().numpy()}")
        print(f"Max activations - Top 10 features: {max_top10.indices.cpu().numpy()}")
        print(f"Sum activations - Top 10 features: {sum_top10.indices.cpu().numpy()}")
        
        # Store for comparison
        activations_by_text[text] = {
            'mean': mean_activations.cpu().numpy(),
            'max': max_activations.cpu().numpy(),
            'sum': sum_activations.cpu().numpy()
        }
    
    # Compare activations between texts
    print(f"\n{'='*60}")
    print("COMPARISON BETWEEN TEXTS")
    print(f"{'='*60}")
    
    texts = list(activations_by_text.keys())
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            text1, text2 = texts[i], texts[j]
            
            # Compare mean activations
            mean1 = activations_by_text[text1]['mean']
            mean2 = activations_by_text[text2]['mean']
            mean_correlation = np.corrcoef(mean1, mean2)[0, 1]
            
            # Compare max activations
            max1 = activations_by_text[text1]['max']
            max2 = activations_by_text[text2]['max']
            max_correlation = np.corrcoef(max1, max2)[0, 1]
            
            # Compare sum activations
            sum1 = activations_by_text[text1]['sum']
            sum2 = activations_by_text[text2]['sum']
            sum_correlation = np.corrcoef(sum1, sum2)[0, 1]
            
            print(f"\nText {i+1} vs Text {j+1}:")
            print(f"  Mean correlation: {mean_correlation:.4f}")
            print(f"  Max correlation:  {max_correlation:.4f}")
            print(f"  Sum correlation:  {sum_correlation:.4f}")
            
            # Check if top features are the same
            mean1_top10 = np.argsort(mean1)[-10:][::-1]
            mean2_top10 = np.argsort(mean2)[-10:][::-1]
            mean_overlap = len(set(mean1_top10) & set(mean2_top10))
            
            max1_top10 = np.argsort(max1)[-10:][::-1]
            max2_top10 = np.argsort(max2)[-10:][::-1]
            max_overlap = len(set(max1_top10) & set(max2_top10))
            
            print(f"  Mean top 10 overlap: {mean_overlap}/10")
            print(f"  Max top 10 overlap:  {max_overlap}/10")

if __name__ == "__main__":
    test_activation_variation()
