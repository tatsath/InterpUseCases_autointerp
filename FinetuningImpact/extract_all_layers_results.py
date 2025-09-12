#!/usr/bin/env python3

import json
import os
from pathlib import Path

def extract_metrics_for_layer(results_dir, run_name, layer, features):
    """Extract metrics for a specific layer and features"""
    metrics = {}
    
    for feature in features:
        score_file_path = os.path.join(results_dir, run_name, "scores", "detection", f"layers.{layer}_latent{feature}.json")
        explanation_file_path = os.path.join(results_dir, run_name, "explanations", f"layers.{layer}_latent{feature}.txt")
        
        f1, precision, recall = 0.0, 0.0, 0.0
        label = "N/A"

        # Extract F1 scores - try both .json and .txt files
        score_file_json = score_file_path
        score_file_txt = score_file_path.replace('.json', '.txt')
        
        if os.path.exists(score_file_json):
            try:
                with open(score_file_json, 'r') as f:
                    score_data = json.load(f)
                    f1 = score_data.get('f1', 0.0)
                    precision = score_data.get('precision', 0.0)
                    recall = score_data.get('recall', 0.0)
            except Exception as e:
                print(f"Error reading JSON score file for feature {feature}: {e}")
        elif os.path.exists(score_file_txt):
            try:
                with open(score_file_txt, 'r') as f:
                    content = f.read()
                    # Try to parse as JSON
                    score_data = json.loads(content)
                    f1 = score_data.get('f1', 0.0)
                    precision = score_data.get('precision', 0.0)
                    recall = score_data.get('recall', 0.0)
            except Exception as e:
                print(f"Error reading TXT score file for feature {feature}: {e}")
                # If it's not JSON, try to extract from the text content
                # Look for F1 score in the text
                if 'f1' in content.lower():
                    try:
                        # Extract F1 score from text
                        import re
                        f1_match = re.search(r'"f1":\s*([0-9.]+)', content)
                        if f1_match:
                            f1 = float(f1_match.group(1))
                    except:
                        pass
        
        # Extract labels
        if os.path.exists(explanation_file_path):
            try:
                with open(explanation_file_path, 'r') as f:
                    label = f.read().strip().replace('"', '')
            except Exception as e:
                print(f"Error reading explanation file for feature {feature}: {e}")

        metrics[feature] = {
            "f1": round(f1, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "label": label
        }
    
    return metrics

def get_features_for_layer(layer):
    """Get the top 10 features for a specific layer from finetuning results"""
    try:
        with open('finetuning_impact_results.json', 'r') as f:
            data = json.load(f)
        features = data[str(layer)]['top_10_improved_features']['feature_indices'][:10]
        return features
    except Exception as e:
        print(f"Error reading features for layer {layer}: {e}")
        return []

def main():
    results_dir = "/home/nvidia/Documents/Hariom/autointerp/autointerp_full/results"
    layers = [4, 10, 16, 22, 28]
    
    all_results = {}
    
    for layer in layers:
        print(f"Processing Layer {layer}...")
        
        # Get features for this layer
        features = get_features_for_layer(layer)
        if not features:
            print(f"No features found for layer {layer}, skipping...")
            continue
        
        print(f"Features for Layer {layer}: {features}")
        
        # Extract base model results
        base_run_name = f"base_model_layer{layer}_all_layers"
        base_metrics = extract_metrics_for_layer(results_dir, base_run_name, layer, features)
        
        # Extract finetuned model results
        finetuned_run_name = f"finetuned_model_layer{layer}_all_layers"
        finetuned_metrics = extract_metrics_for_layer(results_dir, finetuned_run_name, layer, features)
        
        # Store results
        all_results[f"layer_{layer}"] = {
            "features": features,
            "base_model": base_metrics,
            "finetuned_model": finetuned_metrics
        }
        
        print(f"✅ Layer {layer} completed")
    
    # Save comprehensive results
    with open("all_layers_comprehensive_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n🎯 Generating comparison tables...")
    
    # Generate markdown tables for each layer
    for layer in layers:
        if f"layer_{layer}" not in all_results:
            continue
            
        layer_data = all_results[f"layer_{layer}"]
        features = layer_data["features"]
        base_metrics = layer_data["base_model"]
        finetuned_metrics = layer_data["finetuned_model"]
        
        print(f"\n### Layer {layer} Comparison Table")
        print("| Feature | Old Label | New Label | Old F1 | New F1 | F1 Change |")
        print("|---------|-----------|-----------|--------|--------|-----------|")
        
        for feature in features:
            base_data = base_metrics.get(feature, {})
            finetuned_data = finetuned_metrics.get(feature, {})
            
            old_label = base_data.get("label", "N/A")
            new_label = finetuned_data.get("label", "N/A")
            old_f1 = base_data.get("f1", 0.0)
            new_f1 = finetuned_data.get("f1", 0.0)
            f1_change = round(new_f1 - old_f1, 3)
            
            print(f"| {feature} | {old_label} | {new_label} | {old_f1} | {new_f1} | {f1_change:+} |")
    
    print("\n✅ All results extracted and saved to all_layers_comprehensive_results.json")

if __name__ == "__main__":
    main()
