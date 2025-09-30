#!/usr/bin/env python3
"""
Integration script showing how to use the hallucination probe with SAE analysis.
This demonstrates how to combine hallucination detection with feature activation analysis.
"""

import sys
import os
import numpy as np
import joblib
from typing import Dict, List, Tuple

# Add parent directory to path to import from Reply_Tracing_app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_hallucination_probe(probe_path: str = "probe_ckpt/hallu_linear_probe.joblib") -> Dict:
    """Load the trained hallucination probe."""
    if not os.path.exists(probe_path):
        raise FileNotFoundError(f"Probe not found at {probe_path}. Run hallu_probe.py first.")
    
    return joblib.load(probe_path)

def analyze_hallucination_with_sae_features(
    question: str, 
    generated_answer: str,
    sae_features: Dict[int, List[Dict]],  # From your SAE analysis
    probe_ckpt: Dict
) -> Tuple[float, Dict]:
    """
    Analyze hallucination probability and correlate with SAE features.
    
    Args:
        question: The input question
        generated_answer: The model's generated answer
        sae_features: SAE feature activations by layer (from Reply_Tracing_app)
        probe_ckpt: Loaded hallucination probe checkpoint
    
    Returns:
        Tuple of (hallucination_probability, feature_analysis)
    """
    
    # This would need to be implemented to extract features the same way as hallu_probe.py
    # For now, we'll simulate the hallucination probability
    hallucination_prob = np.random.random()  # Replace with actual feature extraction
    
    # Analyze which SAE features are most active when hallucinating
    feature_analysis = {}
    
    for layer, features in sae_features.items():
        # Get top activated features for this layer
        top_features = sorted(features, key=lambda x: x.get('activation', 0), reverse=True)[:5]
        
        feature_analysis[layer] = {
            'top_features': top_features,
            'avg_activation': np.mean([f.get('activation', 0) for f in features]),
            'max_activation': max([f.get('activation', 0) for f in features]) if features else 0,
            'num_high_activation': sum(1 for f in features if f.get('activation', 0) > 1.0)
        }
    
    return hallucination_prob, feature_analysis

def print_analysis_results(question: str, answer: str, hallucination_prob: float, feature_analysis: Dict):
    """Print formatted analysis results."""
    print("=" * 80)
    print("HALLUCINATION ANALYSIS WITH SAE FEATURES")
    print("=" * 80)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"Hallucination Probability: {hallucination_prob:.3f}")
    print(f"Risk Level: {'HIGH' if hallucination_prob > 0.7 else 'MEDIUM' if hallucination_prob > 0.4 else 'LOW'}")
    print()
    
    print("SAE FEATURE ANALYSIS BY LAYER:")
    print("-" * 40)
    
    for layer, analysis in feature_analysis.items():
        print(f"Layer {layer}:")
        print(f"  Average Activation: {analysis['avg_activation']:.3f}")
        print(f"  Max Activation: {analysis['max_activation']:.3f}")
        print(f"  High Activation Features: {analysis['num_high_activation']}")
        print(f"  Top Features:")
        
        for i, feat in enumerate(analysis['top_features'][:3], 1):
            print(f"    {i}. F{feat.get('id', '?')}: {feat.get('activation', 0):.3f} - {feat.get('label', 'Unknown')[:50]}...")
        print()

def main():
    """Example usage of hallucination probe with SAE features."""
    
    print("Loading hallucination probe...")
    try:
        probe_ckpt = load_hallucination_probe()
        print(f"✅ Loaded probe for layer {probe_ckpt['layer_idx']} with {probe_ckpt['pooling']} pooling")
    except FileNotFoundError:
        print("❌ Hallucination probe not found. Please run 'python hallu_probe.py' first.")
        return
    
    # Example SAE features (this would come from your Reply_Tracing_app analysis)
    example_sae_features = {
        4: [
            {"id": 176, "label": "Technology and Innovation", "activation": 1.136},
            {"id": 32, "label": "Punctuation and syntax markers", "activation": 1.028},
            {"id": 347, "label": "Investment advice or guidance", "activation": 0.942}
        ],
        10: [
            {"id": 318, "label": "Symbolic representations of monetary units", "activation": 3.44},
            {"id": 91, "label": "Transitional or explanatory phrases", "activation": 2.453},
            {"id": 162, "label": "Economic growth and inflation trends", "activation": 0.733}
        ],
        20: [
            {"id": 245, "label": "Complex reasoning patterns", "activation": 2.1},
            {"id": 189, "label": "Factual knowledge retrieval", "activation": 1.8},
            {"id": 156, "label": "Confidence indicators", "activation": 1.2}
        ]
    }
    
    # Example analysis
    question = "What are the key factors driving inflation in the current economy?"
    answer = "The key factors include monetary policy, supply chain disruptions, and energy prices."
    
    hallucination_prob, feature_analysis = analyze_hallucination_with_sae_features(
        question, answer, example_sae_features, probe_ckpt
    )
    
    print_analysis_results(question, answer, hallucination_prob, feature_analysis)
    
    print("=" * 80)
    print("INTEGRATION NOTES:")
    print("=" * 80)
    print("1. To fully integrate, implement the extract_features function from hallu_probe.py")
    print("2. Use the same model and tokenizer as your SAE analysis")
    print("3. Extract features at the same layer as the probe was trained on")
    print("4. Correlate high hallucination probability with specific SAE feature patterns")
    print("5. Consider training probes on different layers to find optimal detection points")

if __name__ == "__main__":
    main()
