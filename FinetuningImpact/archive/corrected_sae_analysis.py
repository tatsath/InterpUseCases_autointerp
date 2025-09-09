#!/usr/bin/env python3
"""
Corrected SAE Analysis Script

This script properly analyzes SAE weight changes with the understanding that:
- Each layer has 400 independent features (0-399)
- Feature 205 in Layer 4 is completely different from Feature 205 in Layer 10
- We need to analyze patterns across layers, not assume feature correspondence
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
        
        print(f"    📁 Loading SAE from: {sae_file}")
        
        from safetensors import safe_open
        with safe_open(sae_file, framework="pt", device="cpu") as f:
            # Load encoder weights and bias
            encoder_weight = f.get_tensor('encoder.weight')
            encoder_bias = f.get_tensor('encoder.bias')
            
            print(f"    ✅ Loaded encoder weight shape: {encoder_weight.shape}")
            print(f"    ✅ Loaded encoder bias shape: {encoder_bias.shape}")
            
            return encoder_weight, encoder_bias
            
    except Exception as e:
        print(f"    ❌ Error loading SAE for layer {layer}: {str(e)}")
        return None, None

def analyze_layer_changes(base_weights: torch.Tensor, finetuned_weights: torch.Tensor, 
                         base_bias: torch.Tensor, finetuned_bias: torch.Tensor, layer: int):
    """Analyze changes for a specific layer"""
    
    # Compute weight differences
    weight_diff = finetuned_weights - base_weights
    bias_diff = finetuned_bias - base_bias
    
    # Compute various metrics
    weight_norm_diff = torch.norm(weight_diff, dim=1)  # L2 norm of weight changes per feature
    bias_abs_diff = torch.abs(bias_diff)  # Absolute bias changes
    
    # Find features with largest changes
    top_weight_changes = torch.topk(weight_norm_diff, k=min(20, len(weight_norm_diff)))
    top_bias_changes = torch.topk(bias_abs_diff, k=min(20, len(bias_abs_diff)))
    
    # Compute correlation between weight and bias changes
    weight_bias_correlation = torch.corrcoef(torch.stack([weight_norm_diff, bias_abs_diff]))[0, 1]
    
    return {
        'layer': layer,
        'weight_norm_diff': weight_norm_diff,
        'bias_abs_diff': bias_abs_diff,
        'top_weight_changes': {
            'indices': top_weight_changes.indices.tolist(),
            'values': top_weight_changes.values.tolist()
        },
        'top_bias_changes': {
            'indices': top_bias_changes.indices.tolist(),
            'values': top_bias_changes.values.tolist()
        },
        'weight_bias_correlation': weight_bias_correlation.item() if not torch.isnan(weight_bias_correlation) else 0.0,
        'mean_weight_change': weight_norm_diff.mean().item(),
        'mean_bias_change': bias_abs_diff.mean().item(),
        'std_weight_change': weight_norm_diff.std().item(),
        'std_bias_change': bias_abs_diff.std().item(),
        'max_weight_change': weight_norm_diff.max().item(),
        'max_bias_change': bias_abs_diff.max().item()
    }

def analyze_cross_layer_patterns(layer_results: Dict[int, Dict]):
    """Analyze patterns across layers"""
    
    print("\n🔍 Cross-Layer Pattern Analysis:")
    print("=" * 50)
    
    # Statistical summary across layers
    weight_changes = [data['mean_weight_change'] for data in layer_results.values()]
    bias_changes = [data['mean_bias_change'] for data in layer_results.values()]
    max_weight_changes = [data['max_weight_change'] for data in layer_results.values()]
    max_bias_changes = [data['max_bias_change'] for data in layer_results.values()]
    
    print(f"📊 Statistical Summary Across All Layers:")
    print(f"  • Mean Weight Change: {np.mean(weight_changes):.4f} ± {np.std(weight_changes):.4f}")
    print(f"  • Mean Bias Change: {np.mean(bias_changes):.4f} ± {np.std(bias_changes):.4f}")
    print(f"  • Max Weight Change: {np.max(max_weight_changes):.4f}")
    print(f"  • Max Bias Change: {np.max(max_bias_changes):.4f}")
    
    # Layer-by-layer comparison
    print(f"\n📈 Layer-by-Layer Comparison:")
    print(f"{'Layer':<6} {'Weight Change':<15} {'Bias Change':<15} {'Max Weight':<12} {'Max Bias':<12}")
    print("-" * 70)
    
    for layer in sorted(layer_results.keys()):
        data = layer_results[layer]
        print(f"{layer:<6} {data['mean_weight_change']:<15.4f} {data['mean_bias_change']:<15.4f} "
              f"{data['max_weight_change']:<12.4f} {data['max_bias_change']:<12.4f}")
    
    # Identify layers with most/least changes
    max_weight_layer = max(layer_results.keys(), key=lambda k: layer_results[k]['mean_weight_change'])
    max_bias_layer = max(layer_results.keys(), key=lambda k: layer_results[k]['mean_bias_change'])
    min_weight_layer = min(layer_results.keys(), key=lambda k: layer_results[k]['mean_weight_change'])
    min_bias_layer = min(layer_results.keys(), key=lambda k: layer_results[k]['mean_bias_change'])
    
    print(f"\n🎯 Key Insights:")
    print(f"  • Layer with MAX weight changes: Layer {max_weight_layer} ({layer_results[max_weight_layer]['mean_weight_change']:.4f})")
    print(f"  • Layer with MAX bias changes: Layer {max_bias_layer} ({layer_results[max_bias_layer]['mean_bias_change']:.4f})")
    print(f"  • Layer with MIN weight changes: Layer {min_weight_layer} ({layer_results[min_weight_layer]['mean_weight_change']:.4f})")
    print(f"  • Layer with MIN bias changes: Layer {min_bias_layer} ({layer_results[min_bias_layer]['mean_bias_change']:.4f})")
    
    # Trend analysis
    layers = sorted(layer_results.keys())
    weight_trend = np.polyfit(layers, [layer_results[l]['mean_weight_change'] for l in layers], 1)[0]
    bias_trend = np.polyfit(layers, [layer_results[l]['mean_bias_change'] for l in layers], 1)[0]
    
    print(f"\n📈 Trends Across Layers:")
    print(f"  • Weight change trend: {'Increasing' if weight_trend > 0 else 'Decreasing'} ({weight_trend:.4f} per layer)")
    print(f"  • Bias change trend: {'Increasing' if bias_trend > 0 else 'Decreasing'} ({bias_trend:.4f} per layer)")

def compare_sae_weights_corrected(base_sae_path: str, finetuned_sae_path: str, layers: List[int]):
    """Compare SAE weights with proper understanding of feature independence"""
    
    print("🚀 Corrected SAE Weight Comparison Analysis")
    print("=" * 60)
    print("📋 IMPORTANT: Each layer has 400 independent features (0-399)")
    print("📋 Feature 205 in Layer 4 ≠ Feature 205 in Layer 10")
    print("📋 We analyze patterns across layers, not feature correspondence")
    print("")
    print(f"📋 Base SAE: {base_sae_path}")
    print(f"📋 Finetuned SAE: {finetuned_sae_path}")
    print(f"📋 Layers: {layers}")
    print("")
    
    layer_results = {}
    
    for layer in layers:
        print(f"\n🔍 Analyzing Layer {layer} (Features 0-399)")
        print("-" * 40)
        
        # Load SAE weights
        print("  Loading SAE weights...")
        base_weights, base_bias = load_sae_weights(base_sae_path, layer)
        finetuned_weights, finetuned_bias = load_sae_weights(finetuned_sae_path, layer)
        
        if base_weights is None or finetuned_weights is None:
            print(f"  ❌ Could not load SAE weights for layer {layer}")
            continue
        
        # Analyze changes
        print("  Analyzing weight differences...")
        analysis = analyze_layer_changes(base_weights, finetuned_weights, base_bias, finetuned_bias, layer)
        
        layer_results[layer] = analysis
        
        print(f"  ✅ Layer {layer} analysis complete")
        print(f"    Mean weight change: {analysis['mean_weight_change']:.4f}")
        print(f"    Mean bias change: {analysis['mean_bias_change']:.4f}")
        print(f"    Max weight change: {analysis['max_weight_change']:.4f}")
        print(f"    Max bias change: {analysis['max_bias_change']:.4f}")
        print(f"    Top 5 weight changes: Features {analysis['top_weight_changes']['indices'][:5]}")
        print(f"    Top 5 bias changes: Features {analysis['top_bias_changes']['indices'][:5]}")
    
    # Cross-layer analysis
    analyze_cross_layer_patterns(layer_results)
    
    # Save results
    output_file = "corrected_sae_analysis_results.json"
    
    # Convert tensors to lists for JSON serialization
    json_results = {}
    for layer, data in layer_results.items():
        json_results[layer] = {
            'layer': data['layer'],
            'top_weight_changes': data['top_weight_changes'],
            'top_bias_changes': data['top_bias_changes'],
            'weight_bias_correlation': data['weight_bias_correlation'],
            'mean_weight_change': data['mean_weight_change'],
            'mean_bias_change': data['mean_bias_change'],
            'std_weight_change': data['std_weight_change'],
            'std_bias_change': data['std_bias_change'],
            'max_weight_change': data['max_weight_change'],
            'max_bias_change': data['max_bias_change']
        }
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Print detailed summary
    print("\n📊 Detailed Layer-by-Layer Analysis:")
    print("=" * 60)
    
    for layer, data in layer_results.items():
        print(f"\n🔍 Layer {layer} (Features 0-399):")
        print(f"  📈 Mean Weight Change: {data['mean_weight_change']:.4f} ± {data['std_weight_change']:.4f}")
        print(f"  📈 Mean Bias Change: {data['mean_bias_change']:.4f} ± {data['std_bias_change']:.4f}")
        print(f"  🔗 Weight-Bias Correlation: {data['weight_bias_correlation']:.4f}")
        
        print(f"  🏆 Top 10 Features with Largest Weight Changes:")
        for i, (idx, val) in enumerate(zip(data['top_weight_changes']['indices'][:10], 
                                         data['top_weight_changes']['values'][:10])):
            print(f"    {i+1:2d}. Feature {idx:3d}: {val:.4f}")
        
        print(f"  🎯 Top 10 Features with Largest Bias Changes:")
        for i, (idx, val) in enumerate(zip(data['top_bias_changes']['indices'][:10], 
                                         data['top_bias_changes']['values'][:10])):
            print(f"    {i+1:2d}. Feature {idx:3d}: {val:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Corrected SAE Weight Comparison")
    parser.add_argument("--base_sae", required=True, help="Path to base SAE model")
    parser.add_argument("--finetuned_sae", required=True, help="Path to finetuned SAE model")
    parser.add_argument("--layers", nargs='+', type=int, default=[4, 10, 16, 22, 28], help="Layers to analyze")
    
    args = parser.parse_args()
    
    compare_sae_weights_corrected(
        base_sae_path=args.base_sae,
        finetuned_sae_path=args.finetuned_sae,
        layers=args.layers
    )

if __name__ == "__main__":
    main()
