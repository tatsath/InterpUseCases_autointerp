#!/usr/bin/env python3
"""
SAE Model Comparison Analysis
Compares top 10 features between two SAE models to analyze finetuning impact
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
import json

# Add autointerp to path
sys.path.append('/home/nvidia/Documents/Hariom/autointerp/autointerp_lite')

def load_financial_data():
    """Load financial domain data for activation analysis"""
    financial_file = "/home/nvidia/Documents/Hariom/autointerp/autointerp_lite/financial_texts.txt"
    
    if not os.path.exists(financial_file):
        print(f"❌ Financial data file not found: {financial_file}")
        return None
    
    with open(financial_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"📊 Loaded {len(texts)} financial texts")
    return texts

def analyze_sae_model(model_path: str, model_name: str, layers: List[int], financial_texts: List[str]):
    """Analyze a single SAE model and return top features for each layer"""
    print(f"\n🔍 Analyzing {model_name}")
    print(f"📁 Model path: {model_path}")
    
    results = {}
    
    for layer in layers:
        print(f"\n  📊 Processing Layer {layer}...")
        
        try:
            # Import here to avoid issues if autointerp is not available
            from run_analysis import analyze_layer_features
            
            # Run analysis for this layer
            layer_results = analyze_layer_features(
                base_model="meta-llama/Llama-2-7b-hf",
                sae_model=model_path,
                domain_data=financial_texts,
                layer_idx=layer,
                top_n=10,
                enable_labeling=True,
                labeling_model="Qwen/Qwen2.5-7B-Instruct"
            )
            
            if layer_results is not None:
                results[layer] = layer_results
                print(f"    ✅ Layer {layer}: Found {len(layer_results)} features")
            else:
                print(f"    ❌ Layer {layer}: Analysis failed")
                results[layer] = []
                
        except Exception as e:
            print(f"    ❌ Layer {layer}: Error - {str(e)}")
            results[layer] = []
    
    return results

def compare_features(model1_results: Dict, model2_results: Dict, model1_name: str, model2_name: str):
    """Compare features between two models and identify differences"""
    print(f"\n🔄 Comparing {model1_name} vs {model2_name}")
    
    comparison_results = {}
    
    for layer in [4, 10, 16, 22, 28]:
        print(f"\n  📊 Layer {layer} Comparison:")
        
        if layer not in model1_results or layer not in model2_results:
            print(f"    ❌ Missing data for layer {layer}")
            continue
        
        features1 = model1_results[layer]
        features2 = model2_results[layer]
        
        # Get feature numbers
        features1_nums = set(features1['feature'].tolist() if hasattr(features1, 'feature') else [])
        features2_nums = set(features2['feature'].tolist() if hasattr(features2, 'feature') else [])
        
        # Find differences
        only_in_model1 = features1_nums - features2_nums
        only_in_model2 = features2_nums - features1_nums
        common_features = features1_nums & features2_nums
        
        print(f"    📈 Model 1 unique features: {len(only_in_model1)}")
        print(f"    📈 Model 2 unique features: {len(only_in_model2)}")
        print(f"    📈 Common features: {len(common_features)}")
        
        comparison_results[layer] = {
            'model1_unique': only_in_model1,
            'model2_unique': only_in_model2,
            'common': common_features,
            'model1_features': features1,
            'model2_features': features2
        }
    
    return comparison_results

def generate_comparison_report(comparison_results: Dict, model1_name: str, model2_name: str):
    """Generate detailed comparison report"""
    print(f"\n📋 Generating Comparison Report")
    
    report_data = []
    
    for layer in [4, 10, 16, 22, 28]:
        if layer not in comparison_results:
            continue
        
        layer_data = comparison_results[layer]
        
        # Model 1 unique features
        for feature_num in list(layer_data['model1_unique'])[:10]:  # Top 10
            feature_info = layer_data['model1_features'][
                layer_data['model1_features']['feature'] == feature_num
            ]
            
            if not feature_info.empty:
                report_data.append({
                    'layer': layer,
                    'model': model1_name,
                    'feature_type': 'unique_to_model1',
                    'feature_number': feature_num,
                    'label': feature_info.iloc[0].get('llm_label', 'N/A'),
                    'domain_activation': feature_info.iloc[0].get('domain_activation', 0),
                    'specialization': feature_info.iloc[0].get('specialization', 0),
                    'specialization_conf': feature_info.iloc[0].get('specialization_conf', 0)
                })
        
        # Model 2 unique features
        for feature_num in list(layer_data['model2_unique'])[:10]:  # Top 10
            feature_info = layer_data['model2_features'][
                layer_data['model2_features']['feature'] == feature_num
            ]
            
            if not feature_info.empty:
                report_data.append({
                    'layer': layer,
                    'model': model2_name,
                    'feature_type': 'unique_to_model2',
                    'feature_number': feature_num,
                    'label': feature_info.iloc[0].get('llm_label', 'N/A'),
                    'domain_activation': feature_info.iloc[0].get('domain_activation', 0),
                    'specialization': feature_info.iloc[0].get('specialization', 0),
                    'specialization_conf': feature_info.iloc[0].get('specialization_conf', 0)
                })
    
    return pd.DataFrame(report_data)

def main():
    """Main analysis function"""
    print("🚀 SAE Model Comparison Analysis")
    print("================================")
    
    # Configuration
    model1_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4_10_16_22_28_k32_latents400_wikitext103_torchrun"
    model2_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_finance_layers4_10_16_22_28_k32_latents400_wikitext103_torchrun"
    
    model1_name = "Base Model (Wikitext)"
    model2_name = "Finetuned Model (Finance)"
    
    layers = [4, 10, 16, 22, 28]
    
    # Load financial data
    print("📊 Loading financial domain data...")
    financial_texts = load_financial_data()
    
    if financial_texts is None:
        print("❌ Cannot proceed without financial data")
        return
    
    # Check if models exist
    if not os.path.exists(model1_path):
        print(f"❌ Model 1 not found: {model1_path}")
        return
    
    if not os.path.exists(model2_path):
        print(f"❌ Model 2 not found: {model2_path}")
        return
    
    # Analyze both models
    print(f"\n🔍 Analyzing {model1_name}...")
    model1_results = analyze_sae_model(model1_path, model1_name, layers, financial_texts)
    
    print(f"\n🔍 Analyzing {model2_name}...")
    model2_results = analyze_sae_model(model2_path, model2_name, layers, financial_texts)
    
    # Compare results
    comparison_results = compare_features(model1_results, model2_results, model1_name, model2_name)
    
    # Generate report
    report_df = generate_comparison_report(comparison_results, model1_name, model2_name)
    
    # Save results
    output_file = "sae_model_comparison_report.csv"
    report_df.to_csv(output_file, index=False)
    
    print(f"\n📁 Results saved to: {output_file}")
    
    # Print summary
    print(f"\n📊 Summary Statistics:")
    print(f"Total unique features analyzed: {len(report_df)}")
    
    for layer in layers:
        layer_data = report_df[report_df['layer'] == layer]
        model1_count = len(layer_data[layer_data['model'] == model1_name])
        model2_count = len(layer_data[layer_data['model'] == model2_name])
        print(f"Layer {layer}: {model1_count} unique to {model1_name}, {model2_count} unique to {model2_name}")
    
    print(f"\n✅ Analysis completed!")

if __name__ == "__main__":
    main()
