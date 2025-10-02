#!/usr/bin/env python3
"""
Token-level feature activation analysis for multiple prompts.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open
import os
import pandas as pd

def analyze_token_level_activations():
    """Analyze feature activations at the token level for multiple prompts."""
    
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
    
    # Test prompts
    test_prompts = [
        "The stock market is performing well today with strong gains.",
        "The weather forecast predicts sunny skies and warm temperatures.",
        "Scientists discovered a new species of butterfly in the Amazon rainforest.",
        "The company reported strong quarterly earnings with revenue growth of 15% year-over-year.",
        "Students are preparing for their final exams next week.",
        "The chef prepared a delicious three-course meal for the dinner party."
    ]
    
    print(f"\nAnalyzing token-level activations for {len(test_prompts)} prompts...")
    
    all_results = []
    
    for prompt_idx, text in enumerate(test_prompts):
        print(f"\n{'='*80}")
        print(f"PROMPT {prompt_idx + 1}: {text}")
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
        
        # Compute activations for each token
        feature_activations = torch.matmul(hidden_states, encoder.T) + encoder_bias
        # Shape: (batch_size, seq_len, n_features)
        
        print(f"\nToken-level feature activations:")
        print(f"{'Token':<15} {'Top 5 Features':<50} {'Top 5 Values'}")
        print(f"{'-'*80}")
        
        # Analyze each token
        for token_idx, token in enumerate(tokens):
            # Get activations for this token
            token_activations = feature_activations[0, token_idx, :].cpu().numpy()
            
            # Get top 5 features for this token
            top5_indices = np.argsort(token_activations)[-5:][::-1]
            top5_values = token_activations[top5_indices]
            
            # Format output
            features_str = ", ".join([f"F{idx}" for idx in top5_indices])
            values_str = ", ".join([f"{val:.2f}" for val in top5_values])
            
            print(f"{token:<15} {features_str:<50} {values_str}")
            
            # Store results
            for feat_idx, value in zip(top5_indices, top5_values):
                all_results.append({
                    'prompt_idx': prompt_idx + 1,
                    'prompt': text,
                    'token_idx': token_idx,
                    'token': token,
                    'feature_id': feat_idx,
                    'activation': value,
                    'rank': np.where(top5_indices == feat_idx)[0][0] + 1
                })
    
    # Create summary analysis
    print(f"\n{'='*80}")
    print("SUMMARY ANALYSIS")
    print(f"{'='*80}")
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(all_results)
    
    # Most frequently activated features across all tokens
    print(f"\nMost frequently activated features across all tokens:")
    feature_counts = df['feature_id'].value_counts().head(10)
    for feat_id, count in feature_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  Feature {feat_id:3d}: {count:3d} times ({percentage:5.1f}%)")
    
    # Features that appear in multiple prompts
    print(f"\nFeatures that appear in multiple prompts:")
    prompt_feature_counts = df.groupby(['prompt_idx', 'feature_id']).size().reset_index(name='count')
    multi_prompt_features = prompt_feature_counts.groupby('feature_id')['prompt_idx'].nunique()
    multi_prompt_features = multi_prompt_features[multi_prompt_features > 1].sort_values(ascending=False)
    
    for feat_id, num_prompts in multi_prompt_features.head(10).items():
        print(f"  Feature {feat_id:3d}: appears in {num_prompts} prompts")
    
    # Token-level diversity analysis
    print(f"\nToken-level diversity analysis:")
    for prompt_idx in range(len(test_prompts)):
        prompt_df = df[df['prompt_idx'] == prompt_idx + 1]
        unique_features = prompt_df['feature_id'].nunique()
        total_tokens = prompt_df['token_idx'].nunique()
        print(f"  Prompt {prompt_idx + 1}: {unique_features} unique features across {total_tokens} tokens")
    
    # Save detailed results
    output_file = "token_level_analysis_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    return df

if __name__ == "__main__":
    results = analyze_token_level_activations()
