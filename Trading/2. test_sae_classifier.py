#!/usr/bin/env python3
"""
Test the trained SAE logistic regression classifier
"""

import sys
import os
sys.path.append('/home/nvidia/Documents/Hariom/probetrain')

from sae_logistic_classifier import SAELogisticClassifier

def test_classifier():
    """Test the trained SAE classifier with sample financial news"""
    
    # Initialize classifier
    classifier = SAELogisticClassifier(layer=16, device="cuda:1")
    
    # Load the trained model
    model_dir = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Trading/sae_logistic_results"
    classifier.load_model(model_dir)
    
    # Test with various financial news
    test_texts = [
        "Apple stock surged 8% after strong earnings report",
        "Tesla shares dropped 3% following production delays", 
        "Microsoft stock remained stable during market volatility",
        "Amazon shares jumped 5% on positive revenue forecast",
        "Google stock fell 2% after regulatory concerns",
        "Netflix shares remained unchanged after earnings miss",
        "Meta stock soared 12% on AI breakthrough announcement",
        "Nvidia shares plummeted 8% after chip shortage news",
        "Bitcoin price stabilized around $50,000 level",
        "Oil prices declined 3% on oversupply concerns"
    ]
    
    print("="*80)
    print("SAE LOGISTIC REGRESSION CLASSIFIER - FINANCIAL SENTIMENT TEST")
    print("="*80)
    print(f"Model: Trained on top 40 SAE features from layer 16")
    print(f"Classes: Down (0), Neutral (1), Up (2)")
    print("="*80)
    
    for i, text in enumerate(test_texts, 1):
        result = classifier.predict(text)
        
        print(f"\n{i:2d}. {result['text']}")
        print(f"    Prediction: {result['predicted_class']} ({result['predicted_label']})")
        print(f"    Confidence: {result['confidence']:.3f}")
        print(f"    Probabilities:")
        for class_name, prob in result['class_probabilities'].items():
            print(f"      {class_name}: {prob:.3f}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Count predictions
    predictions = []
    for text in test_texts:
        result = classifier.predict(text)
        predictions.append(result['predicted_class'])
    
    from collections import Counter
    pred_counts = Counter(predictions)
    
    print(f"Total predictions: {len(predictions)}")
    print(f"Down (0): {pred_counts[0]} predictions")
    print(f"Neutral (1): {pred_counts[1]} predictions") 
    print(f"Up (2): {pred_counts[2]} predictions")
    print("="*80)

if __name__ == "__main__":
    test_classifier()
