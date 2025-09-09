#!/usr/bin/env python3

import json
import argparse
import sys
import os
import subprocess
from typing import List, Dict, Any

def run_autointerp_labeling(model_path: str, sae_path: str, feature_idx: int, layer: int, output_dir: str):
    """Run AutoInterp to generate real labels for a specific feature"""
    try:
        print(f"    🏷️ Generating label for Feature {feature_idx} in Layer {layer}...")
        
        # Create output directory for this feature
        feature_output_dir = f"{output_dir}/feature_{feature_idx}_layer_{layer}"
        os.makedirs(feature_output_dir, exist_ok=True)
        
        # Run AutoInterp command
        cmd = [
            "python", "-m", "autointerp_full",
            model_path,
            sae_path,
            "--n_tokens", "1000",  # Smaller for faster processing
            "--feature_num", str(feature_idx),
            "--hookpoints", f"layers.{layer}",
            "--scorers", "detection",
            "--explainer_model", "Qwen/Qwen2.5-7B-Instruct",
            "--explainer_provider", "offline",
            "--explainer_model_max_len", "4096",
            "--num_gpus", "1",
            "--num_examples_per_scorer_prompt", "1",
            "--n_non_activating", "2",
            "--min_examples", "1",
            "--non_activating_source", "FAISS",
            "--faiss_embedding_model", "sentence-transformers/all-MiniLM-L6-v2",
            "--faiss_embedding_cache_dir", ".embedding_cache",
            "--faiss_embedding_cache_enabled",
            "--dataset_repo", "jyanimaulik/yahoo_finance_stockmarket_news",
            "--dataset_name", "default",
            "--dataset_split", "train[:0.1%]",  # Very small sample for speed
            "--filter_bos",
            "--verbose",
            "--name", f"feature_{feature_idx}_layer_{layer}"
        ]
        
        # Change to autointerp directory
        original_dir = os.getcwd()
        os.chdir("../../autointerp/autointerp_full")
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        # Change back to original directory
        os.chdir(original_dir)
        
        if result.returncode == 0:
            # Look for the explanation file
            results_dir = f"../../autointerp/autointerp_full/results/feature_{feature_idx}_layer_{layer}"
            explanation_file = f"{results_dir}/explanations/feature_{feature_idx}.json"
            
            if os.path.exists(explanation_file):
                with open(explanation_file, 'r') as f:
                    explanation_data = json.load(f)
                
                # Extract the explanation text
                explanation = explanation_data.get('explanation', 'No explanation found')
                return explanation
            else:
                print(f"    ⚠️ No explanation file found at {explanation_file}")
                return f"Feature_{feature_idx}_Layer_{layer}_NoExplanation"
        else:
            print(f"    ❌ AutoInterp failed: {result.stderr}")
            return f"Feature_{feature_idx}_Layer_{layer}_Failed"
            
    except subprocess.TimeoutExpired:
        print(f"    ⏰ AutoInterp timed out for Feature {feature_idx}")
        return f"Feature_{feature_idx}_Layer_{layer}_Timeout"
    except Exception as e:
        print(f"    ❌ Error generating label: {str(e)}")
        return f"Feature_{feature_idx}_Layer_{layer}_Error"

def generate_labels_for_model(model_path: str, sae_path: str, feature_indices: List[int], layer: int, output_dir: str):
    """Generate labels for all features of a model"""
    print(f"\n🏷️ Generating labels for {model_path} - Layer {layer}")
    print(f"📋 Features to label: {feature_indices}")
    
    labels = []
    for i, feature_idx in enumerate(feature_indices):
        print(f"  [{i+1}/{len(feature_indices)}] Processing Feature {feature_idx}...")
        label = run_autointerp_labeling(model_path, sae_path, feature_idx, layer, output_dir)
        labels.append(label)
        print(f"    ✅ Label: {label}")
    
    return labels

def main():
    parser = argparse.ArgumentParser(description="Generate real labels using AutoInterp for base and finetuned models")
    parser.add_argument("--base_sae", required=True, help="Path to base SAE model")
    parser.add_argument("--finetuned_sae", required=True, help="Path to finetuned SAE model")
    parser.add_argument("--base_model", required=True, help="Base model path")
    parser.add_argument("--finetuned_model", required=True, help="Finetuned model path")
    parser.add_argument("--layers", nargs="+", type=int, default=[4, 10, 16, 22, 28], help="Layers to analyze")
    parser.add_argument("--top_n", type=int, default=5, help="Number of top features to label (reduced for speed)")
    parser.add_argument("--output_dir", default="real_labels_output", help="Output directory for results")
    
    args = parser.parse_args()
    
    print("🏷️ REAL FEATURE LABELING WITH AUTOINTERP")
    print("=" * 60)
    print(f"📊 Generating real labels for top {args.top_n} features per model per layer")
    print(f"🔍 Analyzing {len(args.layers)} layers: {args.layers}")
    print(f"📁 Output directory: {args.output_dir}")
    print("")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the previous results to get the top features
    try:
        with open("finetuning_impact_results.json", 'r') as f:
            impact_results = json.load(f)
        
        with open("feature_labels_results.json", 'r') as f:
            label_results = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ Required files not found: {e}")
        print("Please run the activation analysis first")
        return
    
    all_results = {}
    
    for layer in args.layers:
        print(f"\n{'='*60}")
        print(f"GENERATING REAL LABELS FOR LAYER {layer}")
        print(f"{'='*60}")
        
        layer_results = {}
        
        # Get top features from the impact analysis
        if str(layer) in impact_results:
            top_improved_features = impact_results[str(layer)]['top_10_improved_features']['feature_indices'][:args.top_n]
            top_finetuned_features = impact_results[str(layer)]['top_10_finetuned_features']['feature_indices'][:args.top_n]
        else:
            print(f"❌ No impact results found for layer {layer}")
            continue
        
        # Get base model features from label results
        if str(layer) in label_results and 'base_model' in label_results[str(layer)]:
            base_features = label_results[str(layer)]['base_model']['features']['feature_indices'][:args.top_n]
        else:
            print(f"❌ No base model features found for layer {layer}")
            continue
        
        # Generate labels for base model
        print(f"\n🔍 Base Model Features: {base_features}")
        base_labels = generate_labels_for_model(
            args.base_model, args.base_sae, base_features, layer, args.output_dir
        )
        
        # Generate labels for finetuned model
        print(f"\n🔍 Finetuned Model Features: {top_finetuned_features}")
        finetuned_labels = generate_labels_for_model(
            args.finetuned_model, args.finetuned_sae, top_finetuned_features, layer, args.output_dir
        )
        
        # Generate labels for top improved features
        print(f"\n🔍 Top Improved Features: {top_improved_features}")
        improved_labels = generate_labels_for_model(
            args.finetuned_model, args.finetuned_sae, top_improved_features, layer, args.output_dir
        )
        
        layer_results = {
            'base_model': {
                'features': base_features,
                'labels': base_labels
            },
            'finetuned_model': {
                'features': top_finetuned_features,
                'labels': finetuned_labels
            },
            'top_improved_features': {
                'features': top_improved_features,
                'labels': improved_labels
            }
        }
        
        all_results[layer] = layer_results
    
    # Save results
    output_file = f"{args.output_dir}/real_labels_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Real labels saved to: {output_file}")
    print("✅ Real feature labeling completed!")

if __name__ == "__main__":
    main()
