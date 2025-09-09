#!/usr/bin/env python3
"""
SAE Weight Comparison Script

This script directly compares SAE encoder weights between base and finetuned models
to identify which features are different and potentially more relevant for financial data.
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
            # Get all keys to see what's available
            keys = f.keys()
            print(f"    📋 Available keys: {list(keys)}")
            
            # Load encoder weights and bias
            encoder_weight = f.get_tensor('encoder.weight')
            encoder_bias = f.get_tensor('encoder.bias')
            
            print(f"    ✅ Loaded encoder weight shape: {encoder_weight.shape}")
            print(f"    ✅ Loaded encoder bias shape: {encoder_bias.shape}")
            
            return encoder_weight, encoder_bias
            
    except Exception as e:
        print(f"    ❌ Error loading SAE for layer {layer}: {str(e)}")
        return None, None

def analyze_weight_differences(base_weights: torch.Tensor, finetuned_weights: torch.Tensor, 
                             base_bias: torch.Tensor, finetuned_bias: torch.Tensor):
    """Analyze differences between base and finetuned SAE weights"""
    
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
        'std_bias_change': bias_abs_diff.std().item()
    }

def compare_sae_weights(base_sae_path: str, finetuned_sae_path: str, layers: List[int]):
    """Compare SAE weights between base and finetuned models"""
    
    print("🚀 SAE Weight Comparison Analysis")
    print("=" * 50)
    print(f"📋 Base SAE: {base_sae_path}")
    print(f"📋 Finetuned SAE: {finetuned_sae_path}")
    print(f"📋 Layers: {layers}")
    print("")
    
    results = {}
    
    for layer in layers:
        print(f"\n🔍 Analyzing Layer {layer}")
        print("-" * 30)
        
        # Load SAE weights
        print("  Loading SAE weights...")
        base_weights, base_bias = load_sae_weights(base_sae_path, layer)
        finetuned_weights, finetuned_bias = load_sae_weights(finetuned_sae_path, layer)
        
        if base_weights is None or finetuned_weights is None:
            print(f"  ❌ Could not load SAE weights for layer {layer}")
            continue
        
        # Analyze differences
        print("  Analyzing weight differences...")
        analysis = analyze_weight_differences(base_weights, finetuned_weights, base_bias, finetuned_bias)
        
        results[layer] = analysis
        
        print(f"  ✅ Layer {layer} analysis complete")
        print(f"    Mean weight change: {analysis['mean_weight_change']:.4f}")
        print(f"    Mean bias change: {analysis['mean_bias_change']:.4f}")
        print(f"    Weight-bias correlation: {analysis['weight_bias_correlation']:.4f}")
        print(f"    Top 5 weight changes: {analysis['top_weight_changes']['indices'][:5]}")
        print(f"    Top 5 bias changes: {analysis['top_bias_changes']['indices'][:5]}")
    
    # Save results
    output_file = "sae_weight_comparison_results.json"
    
    # Convert tensors to lists for JSON serialization
    json_results = {}
    for layer, data in results.items():
        json_results[layer] = {
            'top_weight_changes': data['top_weight_changes'],
            'top_bias_changes': data['top_bias_changes'],
            'weight_bias_correlation': data['weight_bias_correlation'],
            'mean_weight_change': data['mean_weight_change'],
            'mean_bias_change': data['mean_bias_change'],
            'std_weight_change': data['std_weight_change'],
            'std_bias_change': data['std_bias_change']
        }
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Print detailed summary
    print("\n📊 Detailed SAE Weight Comparison Analysis:")
    print("=" * 60)
    
    for layer, data in results.items():
        print(f"\n🔍 Layer {layer}:")
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
    
    # Overall summary
    print(f"\n📊 Overall Summary:")
    print("=" * 30)
    all_weight_changes = [data['mean_weight_change'] for data in results.values()]
    all_bias_changes = [data['mean_bias_change'] for data in results.values()]
    
    print(f"  📈 Average Weight Change Across Layers: {np.mean(all_weight_changes):.4f}")
    print(f"  📈 Average Bias Change Across Layers: {np.mean(all_bias_changes):.4f}")
    print(f"  📊 Layer with Max Weight Change: Layer {max(results.keys(), key=lambda k: results[k]['mean_weight_change'])}")
    print(f"  📊 Layer with Max Bias Change: Layer {max(results.keys(), key=lambda k: results[k]['mean_bias_change'])}")

def main():
    parser = argparse.ArgumentParser(description="SAE Weight Comparison")
    parser.add_argument("--base_sae", required=True, help="Path to base SAE model")
    parser.add_argument("--finetuned_sae", required=True, help="Path to finetuned SAE model")
    parser.add_argument("--layers", nargs='+', type=int, default=[4, 10, 16, 22, 28], help="Layers to analyze")
    
    args = parser.parse_args()
    
    compare_sae_weights(
        base_sae_path=args.base_sae,
        finetuned_sae_path=args.finetuned_sae,
        layers=args.layers
    )

if __name__ == "__main__":
    main()
