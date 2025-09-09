#!/usr/bin/env python3
"""
Simple SAE Feature Comparison

This script directly compares SAE features between base and finetuned models
to identify which features are more activated on financial data after finetuning.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import argparse
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

def load_sae_weights(model_path: str, layer: int):
    """Load SAE encoder weights for a specific layer"""
    try:
        # Try different possible file locations
        possible_paths = [
            f"{model_path}/layers.{layer}/sae.safetensors",
            f"{model_path}/sae_layer_{layer}.pt",
            f"{model_path}/layer_{layer}/sae.pt",
            f"{model_path}/layer_{layer}/model.pt",
            f"{model_path}/layer_{layer}/encoder.pt"
        ]
        
        sae_file = None
        for path in possible_paths:
            if os.path.exists(path):
                sae_file = path
                break
        
        if not sae_file:
            print(f"    ❌ SAE model not found for layer {layer}")
            return None
        
        print(f"    📁 Loading SAE from: {sae_file}")
        
        # Handle different file formats
        if sae_file.endswith('.safetensors'):
            from safetensors import safe_open
            with safe_open(sae_file, framework="pt", device="cpu") as f:
                # Get all keys to see what's available
                keys = f.keys()
                print(f"    📋 Available keys: {list(keys)}")
                
                # Look for encoder weights
                encoder_key = None
                for key in keys:
                    if 'encoder' in key.lower() or 'W_enc' in key.lower():
                        encoder_key = key
                        break
                
                if encoder_key:
                    encoder = f.get_tensor(encoder_key)
                else:
                    # If no encoder key found, try to get the first weight tensor
                    encoder = f.get_tensor(list(keys)[0])
        else:
            sae_data = torch.load(sae_file, map_location='cpu')
            
            # Extract encoder weights
            if isinstance(sae_data, dict):
                encoder = sae_data.get('encoder', sae_data.get('W_enc', sae_data.get('weight')))
                if encoder is None:
                    # Try to find encoder in nested structure
                    for key in sae_data.keys():
                        if 'enc' in key.lower() or 'encoder' in key.lower():
                            encoder = sae_data[key]
                            break
            else:
                # Assume it's a model object
                if hasattr(sae_data, 'encoder'):
                    encoder = sae_data.encoder.weight
                elif hasattr(sae_data, 'weight'):
                    encoder = sae_data.weight
                else:
                    encoder = sae_data
        
        if encoder is None:
            print(f"    ❌ Could not extract encoder weights from {sae_file}")
            return None
        
        print(f"    ✅ Loaded encoder with shape: {encoder.shape}")
        return encoder
        
    except Exception as e:
        print(f"    ❌ Error loading SAE for layer {layer}: {str(e)}")
        return None

def load_financial_data(data_path: str, max_samples: int = 500):
    """Load financial text data"""
    try:
        if data_path.endswith('.txt'):
            with open(data_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f.readlines() if line.strip()]
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            text_col = 'text' if 'text' in df.columns else 'content'
            texts = df[text_col].dropna().astype(str).tolist()
        else:
            print(f"    ❌ Unsupported file format: {data_path}")
            return []
        
        texts = texts[:max_samples]
        print(f"    📊 Loaded {len(texts)} financial text samples")
        return texts
        
    except Exception as e:
        print(f"    ❌ Error loading financial data: {str(e)}")
        return []

def get_model_activations(model_path: str, texts: List[str], layer: int, max_length: int = 256):
    """Get activations from the model for given texts"""
    try:
        print(f"    🔍 Getting activations from {model_path} for layer {layer}")
        
        # Import here to avoid issues if transformers not available
        from transformers import AutoTokenizer, AutoModel
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
        model.eval()
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        all_activations = []
        
        # Process texts in batches
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            if i % 50 == 0:
                print(f"      Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            # Tokenize batch
            inputs = tokenizer(batch_texts, return_tensors="pt", max_length=max_length, 
                             truncation=True, padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                # Get activations for the specified layer
                layer_activations = outputs.hidden_states[layer + 1]  # +1 because layer 0 is embeddings
                all_activations.append(layer_activations.cpu())
        
        # Concatenate all activations
        if all_activations:
            activations = torch.cat(all_activations, dim=0)
            print(f"    ✅ Got activations shape: {activations.shape}")
            return activations
        else:
            return None
            
    except Exception as e:
        print(f"    ❌ Error getting model activations: {str(e)}")
        return None

def compute_feature_activations(activations: torch.Tensor, encoder: torch.Tensor):
    """Compute SAE feature activations"""
    try:
        # Compute SAE activations: ReLU(W_enc @ x + b_enc)
        # For simplicity, assume no bias for now
        sae_activations = torch.relu(torch.matmul(activations, encoder.T))
        return sae_activations
        
    except Exception as e:
        print(f"    ❌ Error computing SAE activations: {str(e)}")
        return None

def compare_sae_features(base_sae_path: str, finetuned_sae_path: str, 
                        base_model_path: str, finetuned_model_path: str,
                        financial_data_path: str, layers: List[int], 
                        top_k: int = 10):
    """Compare SAE features between base and finetuned models"""
    
    print("🚀 Simple SAE Feature Comparison")
    print("=" * 50)
    print(f"📋 Base SAE: {base_sae_path}")
    print(f"📋 Finetuned SAE: {finetuned_sae_path}")
    print(f"📋 Base Model: {base_model_path}")
    print(f"📋 Finetuned Model: {finetuned_model_path}")
    print(f"📋 Financial Data: {financial_data_path}")
    print(f"📋 Layers: {layers}")
    print("")
    
    # Load financial data
    print("📊 Loading financial data...")
    financial_texts = load_financial_data(financial_data_path)
    if not financial_texts:
        print("❌ No financial data loaded")
        return
    
    results = {}
    
    for layer in layers:
        print(f"\n🔍 Analyzing Layer {layer}")
        print("-" * 30)
        
        # Load SAE encoders
        print("  Loading SAE encoders...")
        base_encoder = load_sae_weights(base_sae_path, layer)
        finetuned_encoder = load_sae_weights(finetuned_sae_path, layer)
        
        if base_encoder is None or finetuned_encoder is None:
            print(f"  ❌ Could not load SAE encoders for layer {layer}")
            continue
        
        # Get activations from both models
        print("  Getting model activations...")
        base_activations = get_model_activations(base_model_path, financial_texts, layer)
        finetuned_activations = get_model_activations(finetuned_model_path, financial_texts, layer)
        
        if base_activations is None or finetuned_activations is None:
            print(f"  ❌ Could not get activations for layer {layer}")
            continue
        
        # Compute SAE feature activations
        print("  Computing SAE feature activations...")
        base_feature_acts = compute_feature_activations(base_activations, base_encoder)
        finetuned_feature_acts = compute_feature_activations(finetuned_activations, finetuned_encoder)
        
        if base_feature_acts is None or finetuned_feature_acts is None:
            print(f"  ❌ Could not compute SAE feature activations for layer {layer}")
            continue
        
        # Compute average activations per feature
        base_avg_acts = base_feature_acts.mean(dim=0)
        finetuned_avg_acts = finetuned_feature_acts.mean(dim=0)
        
        # Find features that are more activated in finetuned model
        activation_diff = finetuned_avg_acts - base_avg_acts
        top_improved_features = torch.topk(activation_diff, top_k)
        
        # Find top features for each model
        base_top_features = torch.topk(base_avg_acts, top_k)
        finetuned_top_features = torch.topk(finetuned_avg_acts, top_k)
        
        # Store results
        results[layer] = {
            'base_top_features': {
                'indices': base_top_features.indices.tolist(),
                'values': base_top_features.values.tolist()
            },
            'finetuned_top_features': {
                'indices': finetuned_top_features.indices.tolist(),
                'values': finetuned_top_features.values.tolist()
            },
            'improved_features': {
                'indices': top_improved_features.indices.tolist(),
                'values': top_improved_features.values.tolist()
            },
            'base_avg_activation': base_avg_acts.mean().item(),
            'finetuned_avg_activation': finetuned_avg_acts.mean().item(),
            'activation_improvement': activation_diff.mean().item(),
            'max_improvement': activation_diff.max().item(),
            'num_improved_features': (activation_diff > 0).sum().item()
        }
        
        print(f"  ✅ Layer {layer} analysis complete")
        print(f"    Base avg activation: {base_avg_acts.mean().item():.4f}")
        print(f"    Finetuned avg activation: {finetuned_avg_acts.mean().item():.4f}")
        print(f"    Activation improvement: {activation_diff.mean().item():.4f}")
        print(f"    Max improvement: {activation_diff.max().item():.4f}")
        print(f"    Features improved: {results[layer]['num_improved_features']}/{len(activation_diff)}")
    
    # Save results
    output_file = "simple_sae_comparison_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Print detailed summary
    print("\n📊 Detailed Finetuning Impact Analysis:")
    print("=" * 50)
    for layer, data in results.items():
        print(f"\n🔍 Layer {layer}:")
        print(f"  📈 Overall Activation Improvement: {data['activation_improvement']:.4f}")
        print(f"  🎯 Max Feature Improvement: {data['max_improvement']:.4f}")
        print(f"  📊 Features Improved: {data['num_improved_features']}/{len(data['improved_features']['indices'])}")
        print(f"  🏆 Top 5 Improved Features:")
        for i, (idx, val) in enumerate(zip(data['improved_features']['indices'][:5], 
                                         data['improved_features']['values'][:5])):
            print(f"    {i+1}. Feature {idx}: +{val:.4f}")
        
        print(f"  🔥 Top 5 Finetuned Features:")
        for i, (idx, val) in enumerate(zip(data['finetuned_top_features']['indices'][:5], 
                                         data['finetuned_top_features']['values'][:5])):
            print(f"    {i+1}. Feature {idx}: {val:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Simple SAE Feature Comparison")
    parser.add_argument("--base_sae", required=True, help="Path to base SAE model")
    parser.add_argument("--finetuned_sae", required=True, help="Path to finetuned SAE model")
    parser.add_argument("--base_model", required=True, help="Base Llama model path")
    parser.add_argument("--finetuned_model", required=True, help="Finetuned Llama model path")
    parser.add_argument("--financial_data", required=True, help="Path to financial data")
    parser.add_argument("--layers", nargs='+', type=int, default=[4, 10, 16, 22, 28], help="Layers to analyze")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top features to analyze")
    
    args = parser.parse_args()
    
    compare_sae_features(
        base_sae_path=args.base_sae,
        finetuned_sae_path=args.finetuned_sae,
        base_model_path=args.base_model,
        finetuned_model_path=args.finetuned_model,
        financial_data_path=args.financial_data,
        layers=args.layers,
        top_k=args.top_k
    )

if __name__ == "__main__":
    main()
