#!/usr/bin/env python3
"""
Debug script to check raw feature activations before max aggregation.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open
import os

def debug_raw_activations():
    """Check for always-on features in the Llama SAE model."""
    
    # Load model and tokenizer
    print("Loading Llama 7B model and tokenizer...")
    model_path = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set padding token for Llama tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load SAE weights from local directory
    print("Loading SAE weights from local directory...")
    sae_model_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
    
    # Check what files are available in the local directory
    print("Available files in the local directory:")
    files = os.listdir(sae_model_path)
    for file in files:
        print(f"  {file}")
    
    # Look for layer directories
    layer_dirs = [f for f in files if f.startswith('layers.')]
    print(f"Found layer directories: {layer_dirs}")
    
    if not layer_dirs:
        print("No layer directories found!")
        return
    
    # Analyze multiple layers
    layers_to_check = [4, 10, 16, 22, 28]
    print(f"Analyzing layers: {layers_to_check}")
    
    # Test texts - mix of different content types
    test_texts = [
        "The stock market is performing well today with strong gains in technology stocks",
        "The weather forecast predicts sunny skies and warm temperatures",
        "Scientists discovered a new species of butterfly in the Amazon rainforest",
        "The Lakers won the championship with an incredible performance by LeBron James",
        "The Federal Reserve announced a 0.25% interest rate increase"
    ]
    
    print(f"\nTesting {len(test_texts)} different texts across {len(layers_to_check)} layers...")
    
    # Store results for comparison
    layer_results = {}
    
    for layer_to_check in layers_to_check:
        layer_dir = f"layers.{layer_to_check}"
        
        if layer_dir not in layer_dirs:
            print(f"Layer {layer_to_check} not found! Available layers: {layer_dirs}")
            continue
        
        layer_path = os.path.join(sae_model_path, layer_dir)
        print(f"\n{'='*60}")
        print(f"ANALYZING LAYER {layer_to_check}")
        print(f"{'='*60}")
        
        # Check files in the layer directory
        layer_files = os.listdir(layer_path)
        
        # Look for safetensors file
        safetensors_files = [f for f in layer_files if f.endswith('.safetensors')]
        if not safetensors_files:
            print(f"No safetensors files found in {layer_dir}!")
            continue
        
        sae_file = os.path.join(layer_path, safetensors_files[0])
        print(f"Loading SAE weights from: {sae_file}")
        
        # Load the weights using safetensors
        import safetensors
        sae_data = {}
        with safe_open(sae_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                sae_data[key] = f.get_tensor(key)
    
        # Check what keys are available
        print("Available keys in the SAE file:")
        for key in sae_data.keys():
            print(f"  {key}: {sae_data[key].shape}")
        
        # Try to extract encoder weights
        encoder = None
        encoder_bias = None
        
        # Try different possible key names for safetensors format
        for key in ['encoder.weight', 'W_enc', 'encoder', 'encoder_weights']:
            if key in sae_data:
                encoder = sae_data[key]
                print(f"Found encoder at key: {key}")
                break
        
        for key in ['encoder.bias', 'b_enc', 'encoder_bias']:
            if key in sae_data:
                encoder_bias = sae_data[key]
                print(f"Found encoder bias at key: {key}")
                break
        
        if encoder is None:
            print("Could not find encoder weights!")
            continue
        
        print(f"Encoder shape: {encoder.shape}")
        if encoder_bias is not None:
            print(f"Encoder bias shape: {encoder_bias.shape}")
        else:
            print("No encoder bias found")
        
        # Analyze this layer
        layer_always_on_features = []
        layer_feature_diversity = []
        
        for i, text in enumerate(test_texts):
            print(f"\n--- Text {i+1}: {text[:50]}... ---")
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states_list = outputs.hidden_states
            
            # Check the specified layer
            hidden_states = hidden_states_list[layer_to_check + 1]  # +1 because layer 0 is embeddings
            
            # Move encoder to same device and dtype
            encoder = encoder.to(hidden_states.device).to(hidden_states.dtype)
            encoder_bias = encoder_bias.to(hidden_states.device).to(hidden_states.dtype)
            
            # Compute activations
            feature_activations = torch.matmul(hidden_states, encoder.T) + encoder_bias
            
            # Check for always-on features by counting frequency in top 5
            feature_frequency = {}
            total_tokens = len(tokens)
            
            for token_idx in range(total_tokens):
                token_acts = feature_activations[0, token_idx, :].cpu().numpy()
                top5_indices = np.argsort(token_acts)[-5:][::-1]
                
                for feat_idx in top5_indices:
                    feature_frequency[feat_idx] = feature_frequency.get(feat_idx, 0) + 1
            
            # Sort by frequency
            sorted_features = sorted(feature_frequency.items(), key=lambda x: x[1], reverse=True)
            
            # Check if any features appear in top 5 for >80% of tokens (always-on candidates)
            always_on_candidates = [(feat_idx, count) for feat_idx, count in sorted_features if count / total_tokens > 0.8]
            
            if always_on_candidates:
                print(f"Always-on feature candidates (>80% frequency):")
                for feat_idx, count in always_on_candidates:
                    percentage = (count / total_tokens) * 100
                    print(f"  Feature {feat_idx}: {count}/{total_tokens} tokens ({percentage:.1f}%)")
                    layer_always_on_features.append(feat_idx)
            else:
                print(f"No always-on features detected (no features appear in >80% of tokens)")
            
            # Calculate feature diversity (number of unique features in top 5)
            unique_features = len(set([feat_idx for feat_idx, _ in sorted_features[:10]]))
            layer_feature_diversity.append(unique_features)
            print(f"Feature diversity (unique features in top 10): {unique_features}")
        
        # Store layer results
        layer_results[layer_to_check] = {
            'always_on_features': list(set(layer_always_on_features)),
            'avg_diversity': np.mean(layer_feature_diversity),
            'diversity_scores': layer_feature_diversity
        }
    
    # Print summary comparison
    print(f"\n{'='*80}")
    print("LAYER COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    print(f"{'Layer':<8} {'Always-On Features':<20} {'Avg Diversity':<15} {'Status':<15}")
    print("-" * 80)
    
    for layer, results in layer_results.items():
        always_on_count = len(results['always_on_features'])
        avg_diversity = results['avg_diversity']
        
        if always_on_count == 0:
            status = "✅ Healthy"
        elif always_on_count <= 2:
            status = "⚠️ Mild Issue"
        else:
            status = "❌ Problem"
        
        print(f"{layer:<8} {always_on_count:<20} {avg_diversity:<15.1f} {status:<15}")
    
    print(f"\n{'='*80}")
    print("DETAILED ALWAYS-ON FEATURES BY LAYER")
    print(f"{'='*80}")
    
    for layer, results in layer_results.items():
        if results['always_on_features']:
            print(f"Layer {layer}: Features {results['always_on_features']}")
        else:
            print(f"Layer {layer}: No always-on features detected")

if __name__ == "__main__":
    debug_raw_activations()
