#!/usr/bin/env python3
"""
Interactive SAE Prediction Analyzer
Analyze financial sentiment predictions with feature explanations
"""

import sys
import os
sys.path.append('/home/nvidia/Documents/Hariom/probetrain')

from analyze_sae_prediction import SAEPredictionAnalyzer

def main():
    """Interactive analyzer for SAE predictions"""
    print("="*80)
    print("SAE LOGISTIC REGRESSION FINANCIAL SENTIMENT ANALYZER")
    print("="*80)
    print("Analyzes financial news using top 40 SAE features from layer 16")
    print("Shows prediction probabilities and contributing features with labels")
    print("="*80)
    
    # Initialize analyzer
    analyzer = SAEPredictionAnalyzer(layer=16, device="cuda:1")
    
    # Load model
    model_dir = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Probes/sae_logistic_results"
    print("Loading trained SAE logistic regression model...")
    analyzer.load_model(model_dir)
    print("✅ Model loaded successfully!")
    
    print("\n" + "="*80)
    print("INTERACTIVE ANALYSIS")
    print("="*80)
    print("Enter financial news text to analyze (or 'quit' to exit)")
    print("="*80)
    
    while True:
        print("\n" + "-"*60)
        text = input("Enter financial news text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not text:
            print("Please enter some text to analyze.")
            continue
        
        try:
            # Get number of top features to show
            top_n_input = input("Number of top features to show (default 10): ").strip()
            top_n = int(top_n_input) if top_n_input else 10
            
            # Analyze prediction
            analyzer.print_analysis(text, top_n)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error analyzing text: {e}")
            print("Please try again with different text.")

if __name__ == "__main__":
    main()
