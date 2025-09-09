#!/usr/bin/env python3
"""
Final Finetuning Impact Analysis Script

This script compares two SAE models to analyze the impact of finetuning:
1. Base SAE: llama2_7b_hf_layers4_10_16_22_28_k32_latents400_wikitext103_torchrun (trained on meta-llama/Llama-2-7b-hf)
2. Finetuned SAE: llama2_7b_finance_layers4_10_16_22_28_k32_latents400_wikitext103_torchrun (trained on cxllin/Llama2-7b-Finance)

CRITICAL: Each layer has 400 INDEPENDENT features (0-399)
Feature 205 in Layer 4 ≠ Feature 205 in Layer 10 ≠ Feature 205 in Layer 16

The analysis shows the top 10 features with largest weight changes for each layer independently.
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
from datasets import load_dataset

def load_financial_data(max_samples: int = 100):
    """Load financial data from Yahoo Finance dataset"""
    try:
        print(f"    📊 Loading financial data from jyanimaulik/yahoo_finance_stockmarket_news...")
        dataset = load_dataset("jyanimaulik/yahoo_finance_stockmarket_news", split="train")
        
        # Extract text content
        texts = []
        for item in dataset:
            if 'text' in item and item['text']:
                texts.append(item['text'].strip())
            elif 'content' in item and item['content']:
                texts.append(item['content'].strip())
            elif 'title' in item and item['title']:
                texts.append(item['title'].strip())
        
        # Limit samples
        texts = texts[:max_samples]
        print(f"    ✅ Loaded {len(texts)} financial text samples")
        return texts
        
    except Exception as e:
        print(f"    ❌ Error loading financial data: {str(e)}")
        return []

def get_model_activations(model_path: str, texts: List[str], layer: int, max_length: int = 256):
    """Get activations from the model for given texts"""
    try:
        print(f"    🔍 Getting activations from {model_path} for layer {layer}")
        
        from transformers import AutoTokenizer, AutoModel
        
        # Load tokenizer and model with safetensors to avoid torch version issues
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Try to load with safetensors first, then fallback to regular loading
        try:
            model = AutoModel.from_pretrained(
                model_path, 
                torch_dtype=torch.float16, 
                device_map="auto",
                use_safetensors=True
            )
        except Exception as e:
            print(f"    ⚠️ Safetensors failed, trying regular loading: {str(e)}")
            try:
                model = AutoModel.from_pretrained(
                    model_path, 
                    torch_dtype=torch.float16, 
                    device_map="auto",
                    trust_remote_code=True
                )
            except Exception as e2:
                print(f"    ❌ Both safetensors and regular loading failed: {str(e2)}")
                return None
        
        model.eval()
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        all_activations = []
        
        # Process texts in batches
        batch_size = 2  # Smaller batch size to avoid memory issues
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            if i % 10 == 0:
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

def compute_sae_activations(activations: torch.Tensor, encoder: torch.Tensor, bias: torch.Tensor):
    """Compute SAE feature activations"""
    try:
        # Ensure all tensors have the same dtype
        activations = activations.float()
        encoder = encoder.float()
        bias = bias.float()
        
        # Compute SAE activations: ReLU(W_enc @ x + b_enc)
        # activations shape: [batch_size, seq_len, hidden_dim]
        # encoder shape: [n_features, hidden_dim]
        # bias shape: [n_features]
        
        batch_size, seq_len, hidden_dim = activations.shape
        n_features = encoder.shape[0]
        
        # Reshape activations to [batch_size * seq_len, hidden_dim]
        activations_flat = activations.view(-1, hidden_dim)
        
        # Compute SAE activations
        sae_acts = torch.relu(torch.matmul(activations_flat, encoder.T) + bias)
        
        # Reshape back to [batch_size, seq_len, n_features]
        sae_acts = sae_acts.view(batch_size, seq_len, n_features)
        
        return sae_acts
        
    except Exception as e:
        print(f"    ❌ Error computing SAE activations: {str(e)}")
        return None

def load_sae_weights(model_path: str, layer: int):
    """Load SAE encoder weights for a specific layer"""
    try:
        sae_file = f"{model_path}/layers.{layer}/sae.safetensors"
        
        if not os.path.exists(sae_file):
            print(f"    ❌ SAE model not found for layer {layer}: {sae_file}")
            return None, None
        
        from safetensors import safe_open
        with safe_open(sae_file, framework="pt", device="cpu") as f:
            encoder_weight = f.get_tensor('encoder.weight')
            encoder_bias = f.get_tensor('encoder.bias')
            return encoder_weight, encoder_bias
            
    except Exception as e:
        print(f"    ❌ Error loading SAE for layer {layer}: {str(e)}")
        return None, None

def analyze_layer_activations(base_activations: torch.Tensor, finetuned_activations: torch.Tensor,
                             base_encoder: torch.Tensor, finetuned_encoder: torch.Tensor,
                             base_bias: torch.Tensor, finetuned_bias: torch.Tensor, layer: int):
    """Analyze activation differences for a specific layer on financial data"""
    
    # Compute SAE feature activations
    base_sae_acts = compute_sae_activations(base_activations, base_encoder, base_bias)
    finetuned_sae_acts = compute_sae_activations(finetuned_activations, finetuned_encoder, finetuned_bias)
    
    if base_sae_acts is None or finetuned_sae_acts is None:
        return None
    
    # Compute average activations per feature
    # base_sae_acts shape: [batch_size, seq_len, n_features]
    # We want to average across batch and sequence dimensions
    base_avg_acts = base_sae_acts.mean(dim=(0, 1))  # Average across batch and sequence
    finetuned_avg_acts = finetuned_sae_acts.mean(dim=(0, 1))  # Average across batch and sequence
    
    # Find features that are more activated in finetuned model
    activation_diff = finetuned_avg_acts - base_avg_acts
    top_improved_features = torch.topk(activation_diff, k=10)
    
    # Find top features for each model
    base_top_features = torch.topk(base_avg_acts, k=10)
    finetuned_top_features = torch.topk(finetuned_avg_acts, k=10)
    
    return {
        'layer': layer,
        'top_10_improved_features': {
            'feature_indices': top_improved_features.indices.tolist(),
            'activation_diffs': top_improved_features.values.tolist()
        },
        'top_10_base_features': {
            'feature_indices': base_top_features.indices.tolist(),
            'activation_values': base_top_features.values.tolist()
        },
        'top_10_finetuned_features': {
            'feature_indices': finetuned_top_features.indices.tolist(),
            'activation_values': finetuned_top_features.values.tolist()
        },
        'mean_base_activation': base_avg_acts.mean().item(),
        'mean_finetuned_activation': finetuned_avg_acts.mean().item(),
        'mean_activation_improvement': activation_diff.mean().item(),
        'max_activation_improvement': activation_diff.max().item(),
        'total_features': len(activation_diff)
    }

def print_layer_analysis(layer_data: Dict, layer: int):
    """Print top 10 features for a single layer"""
    
    print(f"\n🔍 LAYER {layer} - TOP 10 FEATURES")
    print("=" * 50)
    print(f"📊 Mean Base Activation: {layer_data['mean_base_activation']:.4f}")
    print(f"📊 Mean Finetuned Activation: {layer_data['mean_finetuned_activation']:.4f}")
    print(f"📈 Mean Activation Improvement: {layer_data['mean_activation_improvement']:.4f}")
    
    print(f"\n🏆 TOP 10 FEATURES WITH LARGEST ACTIVATION IMPROVEMENT:")
    print(f"{'Rank':<4} {'Feature':<8} {'Activation Diff':<15}")
    print("-" * 35)
    
    for i in range(10):  # We know we want top 10
        feature_idx = layer_data['top_10_improved_features']['feature_indices'][i]
        diff_val = layer_data['top_10_improved_features']['activation_diffs'][i]
        print(f"{i+1:<4} {feature_idx:<8} {diff_val:<15.4f}")
    
    print(f"\n🎯 TOP 10 MOST ACTIVATED FEATURES IN FINETUNED MODEL:")
    print(f"{'Rank':<4} {'Feature':<8} {'Activation':<12}")
    print("-" * 30)
    
    for i in range(10):  # We know we want top 10
        feature_idx = layer_data['top_10_finetuned_features']['feature_indices'][i]
        act_val = layer_data['top_10_finetuned_features']['activation_values'][i]
        print(f"{i+1:<4} {feature_idx:<8} {act_val:<12.4f}")

def compare_sae_layers_independently(base_sae_path: str, finetuned_sae_path: str, 
                                   base_model_path: str, finetuned_model_path: str, layers: List[int]):
    """Compare SAE activations on financial data with each layer analyzed completely independently"""
    
    print("🚀 TOP 10 FEATURES PER LAYER - FINETUNING IMPACT")
    print("=" * 60)
    print("📋 Each layer has 400 INDEPENDENT features (0-399)")
    print("📋 Feature 205 in Layer 4 ≠ Feature 205 in Layer 10")
    print("📊 Analyzing activation changes on financial data (functional impact of finetuning)")
    print("")
    
    # Load financial data
    print("📊 Loading financial data...")
    financial_texts = load_financial_data(max_samples=50)  # Use moderate samples for stability
    if not financial_texts:
        print("❌ No financial data loaded")
        return
    
    print(f"✅ Loaded {len(financial_texts)} financial text samples")
    print("")
    
    all_layer_results = {}
    
    for layer in layers:
        print(f"\n{'='*60}")
        print(f"ANALYZING LAYER {layer}")
        print(f"{'='*60}")
        
        # Load SAE weights
        base_encoder, base_bias = load_sae_weights(base_sae_path, layer)
        finetuned_encoder, finetuned_bias = load_sae_weights(finetuned_sae_path, layer)
        
        if base_encoder is None or finetuned_encoder is None:
            print(f"❌ Could not load SAE weights for layer {layer}")
            continue
        
        # Get model activations on financial data
        print(f"🔍 Getting activations from base model for layer {layer}...")
        base_activations = get_model_activations(base_model_path, financial_texts, layer)
        
        print(f"🔍 Getting activations from finetuned model for layer {layer}...")
        finetuned_activations = get_model_activations(finetuned_model_path, financial_texts, layer)
        
        if base_activations is None or finetuned_activations is None:
            print(f"❌ Could not get model activations for layer {layer}")
            continue
        
        # Analyze activation differences on financial data
        layer_analysis = analyze_layer_activations(base_activations, finetuned_activations,
                                                 base_encoder, finetuned_encoder,
                                                 base_bias, finetuned_bias, layer)
        
        if layer_analysis is None:
            print(f"❌ Could not analyze activations for layer {layer}")
            continue
            
        all_layer_results[layer] = layer_analysis
        
        # Print top 10 features for this layer
        print_layer_analysis(layer_analysis, layer)
    
    # Save results
    output_file = "finetuning_impact_results.json"
    
    # Convert to JSON-serializable format
    json_results = {}
    for layer, data in all_layer_results.items():
        json_results[layer] = {
            'layer': data['layer'],
        'top_10_improved_features': data['top_10_improved_features'],
        'top_10_finetuned_features': data['top_10_finetuned_features'],
            'mean_base_activation': data['mean_base_activation'],
            'mean_finetuned_activation': data['mean_finetuned_activation'],
            'mean_activation_improvement': data['mean_activation_improvement'],
            'max_activation_improvement': data['max_activation_improvement'],
            'total_features': data['total_features']
        }
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print(f"\n✅ Top 10 features analysis complete!")

def main():
    parser = argparse.ArgumentParser(description="Final Finetuning Impact Analysis")
    parser.add_argument("--base_sae", required=True, help="Path to base SAE model")
    parser.add_argument("--finetuned_sae", required=True, help="Path to finetuned SAE model")
    parser.add_argument("--base_model", required=True, help="Base Llama model path")
    parser.add_argument("--finetuned_model", required=True, help="Finetuned Llama model path")
    parser.add_argument("--layers", nargs='+', type=int, default=[4, 10, 16, 22, 28], help="Layers to analyze")
    
    args = parser.parse_args()
    
    compare_sae_layers_independently(
        base_sae_path=args.base_sae,
        finetuned_sae_path=args.finetuned_sae,
        base_model_path=args.base_model,
        finetuned_model_path=args.finetuned_model,
        layers=args.layers
    )

if __name__ == "__main__":
    main()