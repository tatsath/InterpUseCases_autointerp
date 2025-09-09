#!/usr/bin/env python3
"""
Direct SAE Model Comparison Script

This script directly compares two SAE models to identify features that are more activated
on financial data in the finetuned model compared to the base model.

Models:
- Base SAE: llama2_7b_hf_layers4_10_16_22_28_k32_latents400_wikitext103_torchrun
- Finetuned SAE: llama2_7b_finance_layers4_10_16_22_28_k32_latents400_wikitext103_torchrun
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel
import argparse
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

def load_sae_model(model_path: str, layer: int):
    """Load SAE model for a specific layer"""
    try:
        # Look for the SAE model files
        sae_file = f"{model_path}/sae_layer_{layer}.pt"
        if not os.path.exists(sae_file):
            # Try alternative naming
            sae_file = f"{model_path}/layer_{layer}/sae.pt"
            if not os.path.exists(sae_file):
                sae_file = f"{model_path}/layer_{layer}/model.pt"
        
        if not os.path.exists(sae_file):
            print(f"    ❌ SAE model not found for layer {layer} in {model_path}")
            return None, None
        
        print(f"    📁 Loading SAE from: {sae_file}")
        sae_data = torch.load(sae_file, map_location='cpu')
        
        # Extract SAE components
        if isinstance(sae_data, dict):
            encoder = sae_data.get('encoder', sae_data.get('W_enc'))
            decoder = sae_data.get('decoder', sae_data.get('W_dec'))
            bias_enc = sae_data.get('b_enc', sae_data.get('bias_enc'))
            bias_dec = sae_data.get('b_dec', sae_data.get('bias_dec'))
        else:
            # Assume it's a model object
            encoder = sae_data.encoder.weight if hasattr(sae_data, 'encoder') else None
            decoder = sae_data.decoder.weight if hasattr(sae_data, 'decoder') else None
            bias_enc = sae_data.encoder.bias if hasattr(sae_data, 'encoder') else None
            bias_dec = sae_data.decoder.bias if hasattr(sae_data, 'decoder') else None
        
        return encoder, decoder
        
    except Exception as e:
        print(f"    ❌ Error loading SAE for layer {layer}: {str(e)}")
        return None, None

def load_financial_data(data_path: str, max_samples: int = 1000):
    """Load financial text data"""
    try:
        if data_path.endswith('.txt'):
            with open(data_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f.readlines() if line.strip()]
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            # Assume text is in a column named 'text' or 'content'
            text_col = 'text' if 'text' in df.columns else 'content'
            texts = df[text_col].dropna().astype(str).tolist()
        else:
            print(f"    ❌ Unsupported file format: {data_path}")
            return []
        
        # Limit samples
        texts = texts[:max_samples]
        print(f"    📊 Loaded {len(texts)} financial text samples")
        return texts
        
    except Exception as e:
        print(f"    ❌ Error loading financial data: {str(e)}")
        return []

def get_model_activations(base_model_path: str, texts: List[str], layer: int, max_length: int = 512):
    """Get activations from the base model for given texts"""
    try:
        print(f"    🔍 Getting activations from base model for layer {layer}")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        model = AutoModel.from_pretrained(base_model_path, torch_dtype=torch.float16)
        model.eval()
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        all_activations = []
        
        for i, text in enumerate(texts):
            if i % 100 == 0:
                print(f"      Processing text {i+1}/{len(texts)}")
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", max_length=max_length, 
                             truncation=True, padding=True)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                # Get activations for the specified layer
                layer_activations = outputs.hidden_states[layer + 1]  # +1 because layer 0 is embeddings
                all_activations.append(layer_activations)
        
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

def compute_sae_activations(activations: torch.Tensor, encoder: torch.Tensor, bias_enc: Optional[torch.Tensor] = None):
    """Compute SAE activations"""
    try:
        # Move to same device
        if encoder.device != activations.device:
            encoder = encoder.to(activations.device)
        if bias_enc is not None and bias_enc.device != activations.device:
            bias_enc = bias_enc.to(activations.device)
        
        # Compute SAE activations: ReLU(W_enc @ x + b_enc)
        if bias_enc is not None:
            sae_activations = torch.relu(torch.matmul(activations, encoder.T) + bias_enc)
        else:
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
    
    print("🚀 Direct SAE Feature Comparison")
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
        
        # Load SAE models
        print("  Loading SAE models...")
        base_encoder, base_decoder = load_sae_model(base_sae_path, layer)
        finetuned_encoder, finetuned_decoder = load_sae_model(finetuned_sae_path, layer)
        
        if base_encoder is None or finetuned_encoder is None:
            print(f"  ❌ Could not load SAE models for layer {layer}")
            continue
        
        # Get activations from both models
        print("  Getting model activations...")
        base_activations = get_model_activations(base_model_path, financial_texts, layer)
        finetuned_activations = get_model_activations(finetuned_model_path, financial_texts, layer)
        
        if base_activations is None or finetuned_activations is None:
            print(f"  ❌ Could not get activations for layer {layer}")
            continue
        
        # Compute SAE activations
        print("  Computing SAE activations...")
        base_sae_acts = compute_sae_activations(base_activations, base_encoder)
        finetuned_sae_acts = compute_sae_activations(finetuned_activations, finetuned_encoder)
        
        if base_sae_acts is None or finetuned_sae_acts is None:
            print(f"  ❌ Could not compute SAE activations for layer {layer}")
            continue
        
        # Compute average activations per feature
        base_avg_acts = base_sae_acts.mean(dim=0)
        finetuned_avg_acts = finetuned_sae_acts.mean(dim=0)
        
        # Find top features for each model
        base_top_features = torch.topk(base_avg_acts, top_k)
        finetuned_top_features = torch.topk(finetuned_avg_acts, top_k)
        
        # Compare activations
        activation_diff = finetuned_avg_acts - base_avg_acts
        top_improved_features = torch.topk(activation_diff, top_k)
        
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
            'activation_improvement': activation_diff.mean().item()
        }
        
        print(f"  ✅ Layer {layer} analysis complete")
        print(f"    Base avg activation: {base_avg_acts.mean().item():.4f}")
        print(f"    Finetuned avg activation: {finetuned_avg_acts.mean().item():.4f}")
        print(f"    Activation improvement: {activation_diff.mean().item():.4f}")
    
    # Save results
    output_file = "sae_comparison_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Print summary
    print("\n📊 Summary of Finetuning Impact:")
    print("=" * 40)
    for layer, data in results.items():
        print(f"\nLayer {layer}:")
        print(f"  Activation Improvement: {data['activation_improvement']:.4f}")
        print(f"  Top 5 Improved Features: {data['improved_features']['indices'][:5]}")
        print(f"  Improvement Values: {[f'{v:.4f}' for v in data['improved_features']['values'][:5]]}")

def main():
    parser = argparse.ArgumentParser(description="Direct SAE Feature Comparison")
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
