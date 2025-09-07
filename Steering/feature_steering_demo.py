#!/usr/bin/env python3
"""
Feature Steering Demo for AutoInterp Analysis

This module demonstrates how to use feature steering to influence model behavior
by manipulating specific features identified through interpretability analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
import os
from pathlib import Path

class FeatureSteering:
    """
    A class for steering model behavior by manipulating specific features.
    """
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize the FeatureSteering class.
        
        Args:
            model_path: Path to the model
            device: Device to run analysis on
        """
        self.model_path = model_path
        self.device = device
        self.steering_results = {}
        self.feature_effects = {}
        
    def load_model(self):
        """Load the model for steering experiments."""
        print(f"Loading model from {self.model_path}")
        # Placeholder for model loading logic
        
    def identify_steerable_features(self, layer_idx: int) -> List[int]:
        """
        Identify features that are good candidates for steering.
        
        Args:
            layer_idx: Index of the layer to analyze
            
        Returns:
            List of feature indices that are steerable
        """
        # Based on the financial analysis, these are key steerable features
        steerable_features = {
            4: [1, 127, 141, 384],  # Basic financial processing
            10: [17, 173, 273, 372],  # Risk and sentiment
            16: [105, 133, 214, 332],  # Advanced operations
            22: [101, 239, 387, 396],  # Quantitative modeling
            28: [154, 171, 333, 389]   # Strategic planning
        }
        
        return steerable_features.get(layer_idx, [])
    
    def apply_feature_steering(self, layer_idx: int, feature_idx: int, 
                             steering_strength: float, input_text: str) -> Dict:
        """
        Apply feature steering to influence model behavior.
        
        Args:
            layer_idx: Index of the layer to steer
            feature_idx: Index of the feature to manipulate
            steering_strength: Strength of the steering (0.0 to 1.0)
            input_text: Input text to process
            
        Returns:
            Dictionary containing steering results
        """
        print(f"Applying feature steering: Layer {layer_idx}, Feature {feature_idx}, Strength {steering_strength}")
        
        # Placeholder for feature steering logic
        steering_result = {
            "layer": layer_idx,
            "feature": feature_idx,
            "steering_strength": steering_strength,
            "input_text": input_text,
            "original_output": "Original model output",
            "steered_output": "Steered model output",
            "effect_size": steering_strength * 0.8,  # Placeholder
            "confidence_change": steering_strength * 0.3
        }
        
        return steering_result
    
    def financial_risk_steering(self, input_text: str, risk_level: str = "high") -> Dict:
        """
        Demonstrate steering for financial risk analysis.
        
        Args:
            input_text: Financial text to analyze
            risk_level: Desired risk level ("low", "medium", "high")
            
        Returns:
            Dictionary containing risk steering results
        """
        print(f"Applying financial risk steering with level: {risk_level}")
        
        # Map risk levels to steering parameters
        risk_steering = {
            "low": {"layer": 10, "feature": 17, "strength": -0.8},  # Reduce risk perception
            "medium": {"layer": 10, "feature": 17, "strength": 0.0},  # Neutral
            "high": {"layer": 10, "feature": 17, "strength": 0.8}   # Increase risk perception
        }
        
        params = risk_steering[risk_level]
        result = self.apply_feature_steering(
            params["layer"], 
            params["feature"], 
            params["strength"], 
            input_text
        )
        
        result["risk_level"] = risk_level
        result["steering_type"] = "financial_risk"
        
        return result
    
    def market_sentiment_steering(self, input_text: str, sentiment: str = "bullish") -> Dict:
        """
        Demonstrate steering for market sentiment analysis.
        
        Args:
            input_text: Market text to analyze
            sentiment: Desired sentiment ("bearish", "neutral", "bullish")
            
        Returns:
            Dictionary containing sentiment steering results
        """
        print(f"Applying market sentiment steering with sentiment: {sentiment}")
        
        # Map sentiment to steering parameters
        sentiment_steering = {
            "bearish": {"layer": 10, "feature": 173, "strength": -0.7},
            "neutral": {"layer": 10, "feature": 173, "strength": 0.0},
            "bullish": {"layer": 10, "feature": 173, "strength": 0.7}
        }
        
        params = sentiment_steering[sentiment]
        result = self.apply_feature_steering(
            params["layer"], 
            params["feature"], 
            params["strength"], 
            input_text
        )
        
        result["sentiment"] = sentiment
        result["steering_type"] = "market_sentiment"
        
        return result
    
    def investment_decision_steering(self, input_text: str, decision_bias: str = "conservative") -> Dict:
        """
        Demonstrate steering for investment decision making.
        
        Args:
            input_text: Investment scenario text
            decision_bias: Desired decision bias ("conservative", "moderate", "aggressive")
            
        Returns:
            Dictionary containing decision steering results
        """
        print(f"Applying investment decision steering with bias: {decision_bias}")
        
        # Map decision bias to steering parameters
        decision_steering = {
            "conservative": {"layer": 28, "feature": 384, "strength": -0.6},
            "moderate": {"layer": 28, "feature": 384, "strength": 0.0},
            "aggressive": {"layer": 28, "feature": 384, "strength": 0.6}
        }
        
        params = decision_steering[decision_bias]
        result = self.apply_feature_steering(
            params["layer"], 
            params["feature"], 
            params["strength"], 
            input_text
        )
        
        result["decision_bias"] = decision_bias
        result["steering_type"] = "investment_decision"
        
        return result
    
    def compare_steering_effects(self, input_text: str) -> Dict:
        """
        Compare the effects of different steering approaches.
        
        Args:
            input_text: Text to analyze with different steering approaches
            
        Returns:
            Dictionary containing comparison results
        """
        print("Comparing different steering effects...")
        
        # Test different steering approaches
        results = {}
        
        # Risk steering comparison
        for risk_level in ["low", "medium", "high"]:
            results[f"risk_{risk_level}"] = self.financial_risk_steering(input_text, risk_level)
        
        # Sentiment steering comparison
        for sentiment in ["bearish", "neutral", "bullish"]:
            results[f"sentiment_{sentiment}"] = self.market_sentiment_steering(input_text, sentiment)
        
        # Decision steering comparison
        for bias in ["conservative", "moderate", "aggressive"]:
            results[f"decision_{bias}"] = self.investment_decision_steering(input_text, bias)
        
        return results
    
    def visualize_steering_effects(self, results: Dict, save_path: str):
        """
        Create visualizations of steering effects.
        
        Args:
            results: Dictionary containing steering results
            save_path: Path to save the visualization
        """
        print(f"Creating steering effects visualization...")
        
        # Extract data for visualization
        steering_types = []
        effect_sizes = []
        confidence_changes = []
        
        for key, result in results.items():
            steering_types.append(result.get("steering_type", "unknown"))
            effect_sizes.append(result.get("effect_size", 0))
            confidence_changes.append(result.get("confidence_change", 0))
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Effect sizes
        ax1.bar(range(len(effect_sizes)), effect_sizes)
        ax1.set_title("Feature Steering Effect Sizes")
        ax1.set_xlabel("Steering Configuration")
        ax1.set_ylabel("Effect Size")
        ax1.set_xticks(range(len(steering_types)))
        ax1.set_xticklabels(steering_types, rotation=45)
        
        # Confidence changes
        ax2.bar(range(len(confidence_changes)), confidence_changes)
        ax2.set_title("Confidence Changes from Steering")
        ax2.set_xlabel("Steering Configuration")
        ax2.set_ylabel("Confidence Change")
        ax2.set_xticks(range(len(steering_types)))
        ax2.set_xticklabels(steering_types, rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_steering_report(self, results: Dict, output_path: str):
        """
        Generate a comprehensive steering analysis report.
        
        Args:
            results: Dictionary containing steering results
            output_path: Path to save the report
        """
        print(f"Generating steering report to {output_path}")
        
        report = {
            "steering_analysis": results,
            "summary": {
                "total_experiments": len(results),
                "steering_types": list(set([r.get("steering_type", "unknown") for r in results.values()])),
                "average_effect_size": np.mean([r.get("effect_size", 0) for r in results.values()]),
                "average_confidence_change": np.mean([r.get("confidence_change", 0) for r in results.values()])
            },
            "recommendations": [
                "Use risk steering for risk assessment applications",
                "Apply sentiment steering for market analysis",
                "Leverage decision steering for investment guidance",
                "Monitor steering effects to ensure desired outcomes"
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

def main():
    """Main function for feature steering demonstration."""
    steerer = FeatureSteering("path/to/model")
    
    # Example financial text
    financial_text = """
    The company's quarterly earnings show strong growth with revenue increasing by 15% 
    year-over-year. However, market volatility remains high due to economic uncertainty. 
    The stock price has been fluctuating between $45 and $55 over the past month.
    """
    
    print("=== Feature Steering Demo ===")
    print(f"Input Text: {financial_text}")
    print()
    
    # Compare different steering effects
    results = steerer.compare_steering_effects(financial_text)
    
    # Generate visualizations
    steerer.visualize_steering_effects(results, "steering_effects_comparison.png")
    
    # Generate report
    steerer.generate_steering_report(results, "feature_steering_report.json")
    
    # Print summary
    print("=== Steering Results Summary ===")
    for key, result in results.items():
        print(f"{key}: Effect Size = {result.get('effect_size', 0):.3f}, "
              f"Confidence Change = {result.get('confidence_change', 0):.3f}")
    
    print("\nSteering demonstration complete!")
    print("Check 'steering_effects_comparison.png' for visualizations")
    print("Check 'feature_steering_report.json' for detailed report")

if __name__ == "__main__":
    main()
