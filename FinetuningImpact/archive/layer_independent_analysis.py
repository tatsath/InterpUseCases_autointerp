#!/usr/bin/env python3
"""
Layer-Independent SAE Analysis

This script analyzes each layer completely independently, showing the top 10 features
for each layer with their specific change values. Each layer has 400 independent features.
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

def analyze_layer_independently(base_weights: torch.Tensor, finetuned_weights: torch.Tensor, 
                               base_bias: torch.Tensor, finetuned_bias: torch.Tensor, layer: int):
    """Analyze changes for a specific layer independently"""
    
    # Compute weight differences
    weight_diff = finetuned_weights - base_weights
    bias_diff = finetuned_bias - base_bias
    
    # Compute various metrics
    weight_norm_diff = torch.norm(weight_diff, dim=1)  # L2 norm of weight changes per feature
    bias_abs_diff = torch.abs(bias_diff)  # Absolute bias changes
    
    # Find top 10 features with largest changes
    top_weight_changes = torch.topk(weight_norm_diff, k=10)
    top_bias_changes = torch.topk(bias_abs_diff, k=10)
    
    # Compute correlation between weight and bias changes
    weight_bias_correlation = torch.corrcoef(torch.stack([weight_norm_diff, bias_abs_diff]))[0, 1]
    
    return {
        'layer': layer,
        'top_10_weight_changes': {
            'feature_indices': top_weight_changes.indices.tolist(),
            'change_values': top_weight_changes.values.tolist()
        },
        'top_10_bias_changes': {
            'feature_indices': top_bias_changes.indices.tolist(),
            'change_values': top_bias_changes.values.tolist()
        },
        'weight_bias_correlation': weight_bias_correlation.item() if not torch.isnan(weight_bias_correlation) else 0.0,
        'mean_weight_change': weight_norm_diff.mean().item(),
        'mean_bias_change': bias_abs_diff.mean().item(),
        'std_weight_change': weight_norm_diff.std().item(),
        'std_bias_change': bias_abs_diff.std().item(),
        'max_weight_change': weight_norm_diff.max().item(),
        'max_bias_change': bias_abs_diff.max().item(),
        'total_features': len(weight_norm_diff)
    }

def print_layer_analysis(layer_data: Dict, layer: int):
    """Print detailed analysis for a single layer"""
    
    print(f"\n🔍 LAYER {layer} - INDEPENDENT ANALYSIS")
    print("=" * 50)
    print(f"📊 Total Features in Layer {layer}: {layer_data['total_features']}")
    print(f"📈 Mean Weight Change: {layer_data['mean_weight_change']:.4f} ± {layer_data['std_weight_change']:.4f}")
    print(f"📈 Mean Bias Change: {layer_data['mean_bias_change']:.4f} ± {layer_data['std_bias_change']:.4f}")
    print(f"🔗 Weight-Bias Correlation: {layer_data['weight_bias_correlation']:.4f}")
    print(f"🎯 Max Weight Change: {layer_data['max_weight_change']:.4f}")
    print(f"🎯 Max Bias Change: {layer_data['max_bias_change']:.4f}")
    
    print(f"\n🏆 TOP 10 FEATURES WITH LARGEST WEIGHT CHANGES:")
    print(f"{'Rank':<4} {'Feature':<8} {'Change Value':<12} {'Change %':<10}")
    print("-" * 40)
    
    for i, (feature_idx, change_val) in enumerate(zip(layer_data['top_10_weight_changes']['feature_indices'], 
                                                     layer_data['top_10_weight_changes']['change_values'])):
        change_percent = (change_val / layer_data['max_weight_change']) * 100
        print(f"{i+1:<4} {feature_idx:<8} {change_val:<12.4f} {change_percent:<10.1f}%")
    
    print(f"\n🎯 TOP 10 FEATURES WITH LARGEST BIAS CHANGES:")
    print(f"{'Rank':<4} {'Feature':<8} {'Change Value':<12} {'Change %':<10}")
    print("-" * 40)
    
    for i, (feature_idx, change_val) in enumerate(zip(layer_data['top_10_bias_changes']['feature_indices'], 
                                                     layer_data['top_10_bias_changes']['change_values'])):
        change_percent = (change_val / layer_data['max_bias_change']) * 100
        print(f"{i+1:<4} {feature_idx:<8} {change_val:<12.4f} {change_percent:<10.1f}%")
    
    # Show which features appear in both top lists
    weight_features = set(layer_data['top_10_weight_changes']['feature_indices'])
    bias_features = set(layer_data['top_10_bias_changes']['feature_indices'])
    common_features = weight_features.intersection(bias_features)
    
    if common_features:
        print(f"\n🔄 FEATURES IN BOTH TOP 10 LISTS: {sorted(common_features)}")
    else:
        print(f"\n🔄 NO FEATURES APPEAR IN BOTH TOP 10 LISTS")

def compare_sae_layers_independently(base_sae_path: str, finetuned_sae_path: str, layers: List[int]):
    """Compare SAE weights with each layer analyzed completely independently"""
    
    print("🚀 LAYER-INDEPENDENT SAE ANALYSIS")
    print("=" * 60)
    print("📋 CRITICAL: Each layer has 400 INDEPENDENT features (0-399)")
    print("📋 Feature 205 in Layer 4 ≠ Feature 205 in Layer 10 ≠ Feature 205 in Layer 16")
    print("📋 We analyze each layer completely separately")
    print("")
    print(f"📋 Base SAE: {base_sae_path}")
    print(f"📋 Finetuned SAE: {finetuned_sae_path}")
    print(f"📋 Layers to analyze: {layers}")
    print("")
    
    all_layer_results = {}
    
    for layer in layers:
        print(f"\n{'='*60}")
        print(f"ANALYZING LAYER {layer}")
        print(f"{'='*60}")
        
        # Load SAE weights
        print(f"Loading SAE weights for Layer {layer}...")
        base_weights, base_bias = load_sae_weights(base_sae_path, layer)
        finetuned_weights, finetuned_bias = load_sae_weights(finetuned_sae_path, layer)
        
        if base_weights is None or finetuned_weights is None:
            print(f"❌ Could not load SAE weights for layer {layer}")
            continue
        
        print(f"✅ Loaded weights: {base_weights.shape}, bias: {base_bias.shape}")
        
        # Analyze changes
        print(f"Analyzing weight differences for Layer {layer}...")
        layer_analysis = analyze_layer_independently(base_weights, finetuned_weights, base_bias, finetuned_bias, layer)
        
        all_layer_results[layer] = layer_analysis
        
        # Print detailed analysis for this layer
        print_layer_analysis(layer_analysis, layer)
    
    # Save results
    output_file = "layer_independent_analysis_results.json"
    
    # Convert to JSON-serializable format
    json_results = {}
    for layer, data in all_layer_results.items():
        json_results[layer] = {
            'layer': data['layer'],
            'top_10_weight_changes': data['top_10_weight_changes'],
            'top_10_bias_changes': data['top_10_bias_changes'],
            'weight_bias_correlation': data['weight_bias_correlation'],
            'mean_weight_change': data['mean_weight_change'],
            'mean_bias_change': data['mean_bias_change'],
            'std_weight_change': data['std_weight_change'],
            'std_bias_change': data['std_bias_change'],
            'max_weight_change': data['max_weight_change'],
            'max_bias_change': data['max_bias_change'],
            'total_features': data['total_features']
        }
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY ACROSS ALL LAYERS")
    print(f"{'='*60}")
    
    # Summary table
    print(f"\n📊 LAYER COMPARISON SUMMARY:")
    print(f"{'Layer':<6} {'Mean Weight':<12} {'Mean Bias':<12} {'Max Weight':<12} {'Max Bias':<12}")
    print("-" * 60)
    
    for layer in sorted(all_layer_results.keys()):
        data = all_layer_results[layer]
        print(f"{layer:<6} {data['mean_weight_change']:<12.4f} {data['mean_bias_change']:<12.4f} "
              f"{data['max_weight_change']:<12.4f} {data['max_bias_change']:<12.4f}")
    
    print(f"\n💾 Results saved to: {output_file}")
    print(f"\n✅ Layer-independent analysis complete!")

def main():
    parser = argparse.ArgumentParser(description="Layer-Independent SAE Analysis")
    parser.add_argument("--base_sae", required=True, help="Path to base SAE model")
    parser.add_argument("--finetuned_sae", required=True, help="Path to finetuned SAE model")
    parser.add_argument("--layers", nargs='+', type=int, default=[4, 10, 16, 22, 28], help="Layers to analyze")
    
    args = parser.parse_args()
    
    compare_sae_layers_independently(
        base_sae_path=args.base_sae,
        finetuned_sae_path=args.finetuned_sae,
        layers=args.layers
    )

if __name__ == "__main__":
    main()
