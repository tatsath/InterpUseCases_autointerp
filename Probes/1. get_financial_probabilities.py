#!/usr/bin/env python3
"""
Get financial sentiment probabilities using the trained probe
"""

import sys
import os
sys.path.append('/home/nvidia/Documents/Hariom/probetrain')

from probetrain.standalone_probe_system import ProbeInvestigator

def get_financial_probabilities(text, probe_dir, layer=0):
    """
    Get financial sentiment probabilities for a given text.
    
    Args:
        text: Input text to analyze
        probe_dir: Path to trained probe directory
        layer: Layer to use for prediction
    
    Returns:
        dict: Prediction results with probabilities
    """
    # Initialize investigator
    investigator = ProbeInvestigator("meta-llama/Llama-2-7b-hf", "cuda")
    investigator.load_model()
    investigator.load_probes(probe_dir, probe_type='multi_class')
    
    # Get prediction
    results = investigator.investigate_sentence(text, layer_indices=[layer], probe_type='multi_class')
    
    # Extract results
    layer_key = f"layer_{layer}"
    if layer_key in results:
        analysis = results[layer_key]
        
        # Get probabilities and prediction
        probabilities = analysis['probabilities']
        prediction = analysis['prediction']
        confidence = analysis['confidence']
        
        # Create class labels
        class_labels = ['Down (0)', 'Neutral (1)', 'Up (2)']
        
        result = {
            'text': text,
            'layer': layer,
            'predicted_class': prediction,
            'predicted_label': class_labels[prediction],
            'confidence': confidence,
            'class_probabilities': {
                class_labels[i]: probabilities[i] for i in range(3)
            },
            'raw_probabilities': probabilities
        }
        
        return result
    else:
        raise ValueError(f"No results found for layer {layer}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python get_financial_probabilities.py 'Your financial news text here'")
        print("Example: python get_financial_probabilities.py 'Apple stock surged 8% after strong earnings'")
        return
    
    text = sys.argv[1]
    probe_dir = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Probes/probetrain_financial_layer16_results"
    
    try:
        result = get_financial_probabilities(text, probe_dir)
        
        print(f"\n{'='*60}")
        print(f"FINANCIAL SENTIMENT PREDICTION")
        print(f"{'='*60}")
        print(f"Text: '{result['text']}'")
        print(f"Layer: {result['layer']}")
        print(f"Predicted Class: {result['predicted_class']} ({result['predicted_label']})")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"\nClass Probabilities:")
        for class_name, prob in result['class_probabilities'].items():
            print(f"  {class_name}: {prob:.3f}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
