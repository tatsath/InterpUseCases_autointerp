#!/usr/bin/env python3
"""
Debug script to check if hidden states are actually different between texts.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def debug_hidden_states():
    """Check if hidden states are actually different between texts."""
    
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
        "The weather forecast predicts sunny skies and warm temperatures", 
        "Scientists discovered a new species of butterfly in the Amazon rainforest"
    ]
    
    print(f"\nTesting {len(test_texts)} different texts...")
    
    hidden_states_by_text = {}
    
    for i, text in enumerate(test_texts):
        print(f"\n--- Text {i+1}: {text[:50]}... ---")
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        print(f"Input IDs shape: {inputs['input_ids'].shape}")
        print(f"Input IDs: {inputs['input_ids'][0][:10]}...")  # First 10 tokens
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[29]  # Layer 28 (0-indexed)
        
        print(f"Hidden states shape: {hidden_states.shape}")
        print(f"Hidden states mean: {hidden_states.mean().item():.6f}")
        print(f"Hidden states std: {hidden_states.std().item():.6f}")
        print(f"Hidden states min: {hidden_states.min().item():.6f}")
        print(f"Hidden states max: {hidden_states.max().item():.6f}")
        
        # Check first few values
        print(f"First 5 hidden state values: {hidden_states[0, 0, :5].cpu().numpy()}")
        
        # Store for comparison
        hidden_states_by_text[text] = hidden_states.cpu().numpy()
    
    # Compare hidden states between texts
    print(f"\n{'='*60}")
    print("COMPARISON OF HIDDEN STATES BETWEEN TEXTS")
    print(f"{'='*60}")
    
    texts = list(hidden_states_by_text.keys())
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            text1, text2 = texts[i], texts[j]
            
            hs1 = hidden_states_by_text[text1]
            hs2 = hidden_states_by_text[text2]
            
            # Check if they're identical
            are_identical = np.array_equal(hs1, hs2)
            print(f"\nText {i+1} vs Text {j+1}:")
            print(f"  Are hidden states identical? {are_identical}")
            
            if not are_identical:
                # Compute correlation
                # Flatten for correlation
                hs1_flat = hs1.flatten()
                hs2_flat = hs2.flatten()
                
                # Make sure they're the same length
                min_len = min(len(hs1_flat), len(hs2_flat))
                hs1_flat = hs1_flat[:min_len]
                hs2_flat = hs2_flat[:min_len]
                
                correlation = np.corrcoef(hs1_flat, hs2_flat)[0, 1]
                print(f"  Correlation: {correlation:.6f}")
                
                # Check differences
                diff = np.abs(hs1_flat - hs2_flat)
                print(f"  Mean absolute difference: {diff.mean():.6f}")
                print(f"  Max absolute difference: {diff.max():.6f}")
            else:
                print("  WARNING: Hidden states are identical!")

if __name__ == "__main__":
    debug_hidden_states()
