#!/usr/bin/env python3
"""
SAE Feature Importance Analysis
Shows which SAE features are most important for financial sentiment
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
import json

# Add current directory to path
sys.path.append('/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Probes')

def analyze_feature_importance():
    """Analyze SAE feature importance"""
    print("="*60)
    print("SAE FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Load model and metadata
    model_path = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Probes/sae_logistic_results/sae_logistic_model.joblib"
    metadata_path = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Probes/sae_logistic_results/model_metadata.json"
    
    model = joblib.load(model_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    feature_indices = metadata['feature_indices']
    
    # Get feature importance (coefficient magnitudes)
    feature_importance = np.abs(model.coef_).mean(axis=0)
    
    # Create feature analysis
    feature_analysis = []
    for i, (feature_id, importance) in enumerate(zip(feature_indices, feature_importance)):
        feature_analysis.append({
            'rank': i + 1,
            'feature_id': feature_id,
            'importance': importance
        })
    
    # Sort by importance
    feature_analysis.sort(key=lambda x: x['importance'], reverse=True)
    
    print(f"Top 20 Most Important SAE Features:")
    print("-" * 40)
    print(f"{'Rank':<4} {'Feature ID':<10} {'Importance':<12}")
    print("-" * 40)
    
    for feature in feature_analysis[:20]:
        print(f"{feature['rank']:<4} {feature['feature_id']:<10} {feature['importance']:<12.4f}")
    
    print("="*60)

if __name__ == "__main__":
    analyze_feature_importance()
