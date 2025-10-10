#!/usr/bin/env python3
"""
Analyze SAE Logistic Regression Predictions
Shows top features responsible for predictions with their labels
"""

import sys
import os
import pandas as pd
import numpy as np
sys.path.append('/home/nvidia/Documents/Hariom/probetrain')

from sae_logistic_classifier import SAELogisticClassifier

class SAEPredictionAnalyzer:
    """Analyze SAE predictions with feature importance and labels"""
    
    def __init__(self, layer=16, device="cuda:1"):
        self.layer = layer
        self.device = device
        self.classifier = None
        self.feature_labels = None
        
    def load_model(self, model_dir):
        """Load the trained classifier"""
        self.classifier = SAELogisticClassifier(layer=self.layer, device=self.device)
        self.classifier.load_model(model_dir)
        
        # Load feature labels from the analysis results
        self._load_feature_labels()
        
    def _load_feature_labels(self):
        """Load feature labels from the combined analysis results"""
        analysis_path = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/FinanceLabeling/combined_analysis_results.xlsx"
        df = pd.read_excel(analysis_path)
        
        # Get layer 16 features
        layer16_features = df[df['layer'] == 16].sort_values('f1_score', ascending=False)
        
        # Create mapping from feature ID to label
        self.feature_labels = {}
        for _, row in layer16_features.iterrows():
            self.feature_labels[row['feature']] = {
                'label': row['label'],
                'f1_score': row['f1_score'],
                'accuracy': row['accuracy']
            }
    
    def analyze_prediction(self, text, top_n=10):
        """Analyze a prediction and show top contributing features"""
        if self.classifier is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Get prediction
        result = self.classifier.predict(text)
        
        # Extract SAE features for this text
        sae_features = self.classifier._extract_sae_features(text)
        selected_features = sae_features[self.classifier.feature_indices]
        
        # Get model coefficients for each class
        coefficients = self.classifier.model.coef_  # [3, 40] for 3 classes and 40 features
        
        # Calculate actual contribution for each feature
        # For multi-class, we need to consider the predicted class
        predicted_class = result['predicted_class']
        class_coefficients = coefficients[predicted_class]  # [40] coefficients for predicted class
        
        # Calculate feature contributions: activation * coefficient
        feature_contributions = selected_features * class_coefficients
        
        # Create feature analysis
        feature_analysis = []
        for i, (feature_id, activation, contribution) in enumerate(zip(
            self.classifier.feature_indices, 
            selected_features, 
            feature_contributions
        )):
            if feature_id in self.feature_labels:
                feature_info = self.feature_labels[feature_id]
                feature_analysis.append({
                    'rank': i + 1,
                    'feature_id': feature_id,
                    'activation': activation,
                    'coefficient': class_coefficients[i],
                    'contribution': contribution,
                    'label': feature_info['label'],
                    'f1_score': feature_info['f1_score'],
                    'accuracy': feature_info['accuracy']
                })
        
        # Sort by actual contribution (not just importance)
        feature_analysis.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        return {
            'prediction': result,
            'top_features': feature_analysis[:top_n],
            'all_features': feature_analysis
        }
    
    def print_analysis(self, text, top_n=10):
        """Print detailed analysis of a prediction"""
        analysis = self.analyze_prediction(text, top_n)
        
        print("="*100)
        print("SAE LOGISTIC REGRESSION PREDICTION ANALYSIS")
        print("="*100)
        print(f"Text: '{text}'")
        print(f"Predicted Class: {analysis['prediction']['predicted_class']} ({analysis['prediction']['predicted_label']})")
        print(f"Confidence: {analysis['prediction']['confidence']:.3f}")
        print(f"Class Probabilities:")
        for class_name, prob in analysis['prediction']['class_probabilities'].items():
            print(f"  {class_name}: {prob:.3f}")
        
        print(f"\nTop {top_n} Contributing Features:")
        print("-" * 120)
        print(f"{'Rank':<4} {'Feature':<8} {'Activation':<12} {'Coefficient':<12} {'Contribution':<12} {'F1 Score':<10} {'Label'}")
        print("-" * 120)
        
        for feature in analysis['top_features']:
            print(f"{feature['rank']:<4} {feature['feature_id']:<8} {feature['activation']:<12.4f} {feature['coefficient']:<12.4f} {feature['contribution']:<12.4f} {feature['f1_score']:<10.3f} {feature['label'][:40]}...")
        
        print("="*100)
        
        return analysis

def main():
    """Main function to analyze predictions"""
    if len(sys.argv) < 2:
        print("Usage: python analyze_sae_prediction.py 'Your financial news text here' [top_n]")
        print("Example: python analyze_sae_prediction.py 'Apple stock surged 8% after strong earnings' 15")
        return
    
    text = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    # Initialize analyzer
    analyzer = SAEPredictionAnalyzer(layer=16, device="cuda:1")
    
    # Load model
    model_dir = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Probes/sae_logistic_results"
    analyzer.load_model(model_dir)
    
    # Analyze prediction
    analyzer.print_analysis(text, top_n)

if __name__ == "__main__":
    main()
