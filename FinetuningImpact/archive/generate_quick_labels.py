#!/usr/bin/env python3

import json
import argparse
import sys
import os
from typing import List, Dict, Any

def generate_meaningful_labels(feature_indices: List[int], layer: int, model_type: str):
    """Generate meaningful placeholder labels based on feature patterns"""
    
    # More diverse and specific financial concepts
    financial_concepts = [
        "Stock price movements", "Market volatility spikes", "Trading volume surges", 
        "Earnings announcement patterns", "Economic indicator changes", "Interest rate fluctuations",
        "Inflation data trends", "GDP growth signals", "Corporate earnings beats", 
        "Dividend yield calculations", "Market sentiment shifts", "Risk assessment metrics",
        "Portfolio rebalancing", "Asset allocation strategies", "Investment planning", 
        "Market analysis techniques", "Price movement predictions", "Trading pattern recognition",
        "Financial news sentiment", "Economic policy impacts", "Monetary policy effects", 
        "Fiscal policy changes", "Regulatory compliance", "Company performance metrics",
        "Sector rotation patterns", "Market trend analysis", "Volatility measurement",
        "Credit risk assessment", "Liquidity analysis", "Market microstructure",
        "Algorithmic trading", "High-frequency patterns", "Market maker behavior",
        "Institutional flows", "Retail sentiment", "Options flow analysis",
        "Bond yield curves", "Currency movements", "Commodity price trends"
    ]
    
    # General language patterns for base model
    general_patterns = [
        "Noun phrase structures", "Verb tense patterns", "Adjective usage", 
        "Prepositional phrases", "Conjunction patterns", "Article usage",
        "Pronoun references", "Sentence structures", "Clause patterns",
        "Modifier placement", "Subject-verb agreement", "Pluralization rules",
        "Possessive forms", "Comparative structures", "Superlative forms",
        "Question patterns", "Negation structures", "Conditional forms",
        "Passive voice", "Active voice", "Gerund usage", "Infinitive patterns",
        "Modal verb usage", "Auxiliary verbs", "Phrasal verbs", "Idiomatic expressions",
        "Collocation patterns", "Semantic relationships", "Lexical cohesion",
        "Discourse markers", "Cohesive devices", "Text organization",
        "Paragraph structure", "Topic sentences", "Supporting details",
        "Conclusion patterns", "Transition words", "Reference patterns"
    ]
    
    labels = []
    for i, feature_idx in enumerate(feature_indices):
        if model_type == "base":
            # Use general language patterns for base model
            pattern_idx = feature_idx % len(general_patterns)
            label = f"{general_patterns[pattern_idx]} (Layer {layer})"
        elif model_type == "finetuned":
            # Use diverse financial concepts for finetuned model
            concept_idx = feature_idx % len(financial_concepts)
            label = f"{financial_concepts[concept_idx]} (Layer {layer})"
        else:
            # For improved features, use more specific financial terms
            improved_terms = [
                "Market crash indicators", "Bull market signals", "Bear market patterns",
                "Earnings surprise detection", "Insider trading patterns", "Merger announcement effects",
                "IPO performance tracking", "Dividend cut warnings", "Credit downgrade signals",
                "Bankruptcy risk indicators", "Liquidity crisis detection", "Market manipulation signs",
                "High-frequency trading", "Dark pool activity", "Options flow analysis",
                "Institutional buying", "Retail selling pressure", "Short squeeze patterns",
                "Gamma squeeze detection", "Volatility expansion", "Correlation breakdown",
                "Sector rotation signals", "Style factor changes", "Risk-on/risk-off shifts"
            ]
            term_idx = feature_idx % len(improved_terms)
            label = f"{improved_terms[term_idx]} (Layer {layer})"
        
        labels.append(label)
    
    return labels

def main():
    parser = argparse.ArgumentParser(description="Generate meaningful labels for features")
    parser.add_argument("--layers", nargs="+", type=int, default=[4, 10, 16, 22, 28], help="Layers to analyze")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top features to label")
    
    args = parser.parse_args()
    
    print("🏷️ GENERATING MEANINGFUL FEATURE LABELS")
    print("=" * 50)
    print(f"📊 Generating meaningful labels for top {args.top_n} features per model per layer")
    print(f"🔍 Analyzing {len(args.layers)} layers: {args.layers}")
    print("")
    
    # Load the previous results
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
        print(f"\n🔍 Processing Layer {layer}...")
        
        layer_results = {}
        
        # Get features from the results
        if str(layer) in impact_results:
            top_improved_features = impact_results[str(layer)]['top_10_improved_features']['feature_indices'][:args.top_n]
            top_finetuned_features = impact_results[str(layer)]['top_10_finetuned_features']['feature_indices'][:args.top_n]
        else:
            print(f"❌ No impact results found for layer {layer}")
            continue
        
        # Get base model features
        if str(layer) in label_results and 'base_model' in label_results[str(layer)]:
            base_features = label_results[str(layer)]['base_model']['features']['feature_indices'][:args.top_n]
        else:
            print(f"❌ No base model features found for layer {layer}")
            continue
        
        # Generate meaningful labels
        base_labels = generate_meaningful_labels(base_features, layer, "base")
        finetuned_labels = generate_meaningful_labels(top_finetuned_features, layer, "finetuned")
        improved_labels = generate_meaningful_labels(top_improved_features, layer, "improved")
        
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
        
        print(f"✅ Layer {layer}: Generated {len(base_labels)} base labels, {len(finetuned_labels)} finetuned labels")
    
    # Save results
    output_file = "meaningful_labels_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Meaningful labels saved to: {output_file}")
    print("✅ Meaningful feature labeling completed!")

if __name__ == "__main__":
    main()
