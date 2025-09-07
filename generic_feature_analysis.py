#!/usr/bin/env python3
"""
Generic Feature Analysis Script for AutoInterp

This script provides generic feature analysis capabilities for
interpreting and analyzing neural network features.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from pathlib import Path
import argparse
from datetime import datetime

class GenericFeatureAnalysis:
    """
    A class for generic feature analysis and interpretation.
    """
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize the GenericFeatureAnalysis class.
        
        Args:
            model_path: Path to the model
            device: Device to run analysis on
        """
        self.model_path = model_path
        self.device = device
        self.feature_data = {}
        self.analysis_results = {}
        
    def load_model(self) -> Any:
        """
        Load the model for feature analysis.
        
        Returns:
            Loaded model
        """
        print(f"Loading model from {self.model_path}")
        
        # Placeholder for model loading logic
        # In a real implementation, this would load the actual model
        model = {
            "name": "generic_model",
            "path": self.model_path,
            "device": self.device,
            "loaded": True
        }
        
        print("Model loaded successfully")
        return model
    
    def extract_features(self, model: Any, input_data: List[str], layer: int) -> Dict:
        """
        Extract features from the model for given input data.
        
        Args:
            model: Loaded model
            input_data: List of input texts
            layer: Layer index to extract features from
            
        Returns:
            Dictionary containing extracted features
        """
        print(f"Extracting features from layer {layer} for {len(input_data)} samples")
        
        # Placeholder for feature extraction
        # In a real implementation, this would extract actual features
        features = {
            "layer": layer,
            "num_samples": len(input_data),
            "feature_dimensions": 512,  # Placeholder
            "features": np.random.randn(len(input_data), 512),  # Placeholder
            "feature_names": [f"feature_{i}" for i in range(512)]
        }
        
        print(f"Extracted {features['feature_dimensions']} features from layer {layer}")
        return features
    
    def analyze_feature_activations(self, features: Dict) -> Dict:
        """
        Analyze feature activation patterns.
        
        Args:
            features: Dictionary containing extracted features
            
        Returns:
            Dictionary containing activation analysis
        """
        print("Analyzing feature activation patterns...")
        
        feature_matrix = features['features']
        layer = features['layer']
        
        # Calculate activation statistics
        activation_stats = {
            "layer": layer,
            "mean_activations": np.mean(feature_matrix, axis=0),
            "std_activations": np.std(feature_matrix, axis=0),
            "max_activations": np.max(feature_matrix, axis=0),
            "min_activations": np.min(feature_matrix, axis=0),
            "sparsity": np.mean(feature_matrix == 0, axis=0)
        }
        
        # Identify top activating features
        mean_activations = activation_stats["mean_activations"]
        top_features = np.argsort(mean_activations)[-10:][::-1]  # Top 10
        
        activation_stats["top_features"] = [
            {
                "feature_id": int(idx),
                "mean_activation": float(mean_activations[idx]),
                "std_activation": float(activation_stats["std_activations"][idx]),
                "max_activation": float(activation_stats["max_activations"][idx])
            }
            for idx in top_features
        ]
        
        print(f"Identified {len(top_features)} top activating features")
        return activation_stats
    
    def analyze_feature_correlations(self, features: Dict) -> Dict:
        """
        Analyze correlations between features.
        
        Args:
            features: Dictionary containing extracted features
            
        Returns:
            Dictionary containing correlation analysis
        """
        print("Analyzing feature correlations...")
        
        feature_matrix = features['features']
        layer = features['layer']
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(feature_matrix.T)
        
        # Find highly correlated feature pairs
        high_correlations = []
        for i in range(len(correlation_matrix)):
            for j in range(i + 1, len(correlation_matrix)):
                corr = correlation_matrix[i, j]
                if abs(corr) > 0.7:  # High correlation threshold
                    high_correlations.append({
                        "feature_1": int(i),
                        "feature_2": int(j),
                        "correlation": float(corr)
                    })
        
        correlation_analysis = {
            "layer": layer,
            "correlation_matrix": correlation_matrix.tolist(),
            "high_correlations": high_correlations,
            "num_high_correlations": len(high_correlations)
        }
        
        print(f"Found {len(high_correlations)} highly correlated feature pairs")
        return correlation_analysis
    
    def analyze_feature_importance(self, features: Dict, labels: List[str] = None) -> Dict:
        """
        Analyze feature importance for classification tasks.
        
        Args:
            features: Dictionary containing extracted features
            labels: Optional labels for supervised analysis
            
        Returns:
            Dictionary containing importance analysis
        """
        print("Analyzing feature importance...")
        
        feature_matrix = features['features']
        layer = features['layer']
        
        # Placeholder for importance analysis
        # In a real implementation, this would use actual importance metrics
        importance_scores = np.random.rand(feature_matrix.shape[1])
        
        # Rank features by importance
        importance_ranking = np.argsort(importance_scores)[::-1]
        
        importance_analysis = {
            "layer": layer,
            "importance_scores": importance_scores.tolist(),
            "top_important_features": [
                {
                    "feature_id": int(idx),
                    "importance_score": float(importance_scores[idx]),
                    "rank": int(rank + 1)
                }
                for rank, idx in enumerate(importance_ranking[:20])  # Top 20
            ]
        }
        
        print(f"Analyzed importance for {len(importance_scores)} features")
        return importance_analysis
    
    def create_feature_visualizations(self, analysis_results: Dict, output_dir: str):
        """
        Create visualizations for feature analysis.
        
        Args:
            analysis_results: Dictionary containing analysis results
            output_dir: Directory to save visualizations
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"Creating feature visualizations in {output_path}")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Activation distribution
        if 'activation_analysis' in analysis_results:
            activation_data = analysis_results['activation_analysis']
            
            plt.figure(figsize=(12, 8))
            
            # Mean activations histogram
            plt.subplot(2, 2, 1)
            plt.hist(activation_data['mean_activations'], bins=50, alpha=0.7)
            plt.title('Distribution of Mean Feature Activations')
            plt.xlabel('Mean Activation')
            plt.ylabel('Frequency')
            
            # Top features bar plot
            plt.subplot(2, 2, 2)
            top_features = activation_data['top_features'][:10]
            feature_ids = [f['feature_id'] for f in top_features]
            mean_activations = [f['mean_activation'] for f in top_features]
            plt.bar(range(len(feature_ids)), mean_activations)
            plt.title('Top 10 Activating Features')
            plt.xlabel('Feature Rank')
            plt.ylabel('Mean Activation')
            plt.xticks(range(len(feature_ids)), feature_ids, rotation=45)
            
            # Activation sparsity
            plt.subplot(2, 2, 3)
            plt.hist(activation_data['sparsity'], bins=50, alpha=0.7)
            plt.title('Feature Sparsity Distribution')
            plt.xlabel('Sparsity (Fraction of Zeros)')
            plt.ylabel('Frequency')
            
            # Activation variance
            plt.subplot(2, 2, 4)
            plt.hist(activation_data['std_activations'], bins=50, alpha=0.7)
            plt.title('Feature Activation Variance')
            plt.xlabel('Standard Deviation')
            plt.ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(output_path / 'activation_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Correlation heatmap
        if 'correlation_analysis' in analysis_results:
            correlation_data = analysis_results['correlation_analysis']
            
            plt.figure(figsize=(10, 8))
            correlation_matrix = np.array(correlation_data['correlation_matrix'])
            
            # Sample a subset for visualization (first 50 features)
            subset_size = min(50, correlation_matrix.shape[0])
            subset_matrix = correlation_matrix[:subset_size, :subset_size]
            
            sns.heatmap(subset_matrix, cmap='coolwarm', center=0, square=True)
            plt.title(f'Feature Correlation Matrix (First {subset_size} Features)')
            plt.xlabel('Feature Index')
            plt.ylabel('Feature Index')
            plt.tight_layout()
            plt.savefig(output_path / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Importance analysis
        if 'importance_analysis' in analysis_results:
            importance_data = analysis_results['importance_analysis']
            
            plt.figure(figsize=(12, 6))
            
            # Importance scores distribution
            plt.subplot(1, 2, 1)
            plt.hist(importance_data['importance_scores'], bins=50, alpha=0.7)
            plt.title('Distribution of Feature Importance Scores')
            plt.xlabel('Importance Score')
            plt.ylabel('Frequency')
            
            # Top important features
            plt.subplot(1, 2, 2)
            top_important = importance_data['top_important_features'][:15]
            feature_ids = [f['feature_id'] for f in top_important]
            importance_scores = [f['importance_score'] for f in top_important]
            plt.bar(range(len(feature_ids)), importance_scores)
            plt.title('Top 15 Most Important Features')
            plt.xlabel('Feature Rank')
            plt.ylabel('Importance Score')
            plt.xticks(range(len(feature_ids)), feature_ids, rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_path / 'importance_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_analysis_report(self, analysis_results: Dict, output_dir: str):
        """
        Generate a comprehensive analysis report.
        
        Args:
            analysis_results: Dictionary containing analysis results
            output_dir: Directory to save the report
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"Generating analysis report in {output_path}")
        
        # Create summary statistics
        summary = {
            "analysis_timestamp": datetime.now().isoformat(),
            "model_path": self.model_path,
            "device": self.device,
            "layers_analyzed": list(analysis_results.keys()),
            "total_features_analyzed": sum(
                len(results.get('features', {}).get('feature_names', []))
                for results in analysis_results.values()
            )
        }
        
        # Add layer-specific summaries
        layer_summaries = {}
        for layer, results in analysis_results.items():
            layer_summary = {
                "layer": layer,
                "num_features": len(results.get('features', {}).get('feature_names', [])),
                "num_samples": results.get('features', {}).get('num_samples', 0)
            }
            
            if 'activation_analysis' in results:
                activation_data = results['activation_analysis']
                layer_summary.update({
                    "mean_activation": float(np.mean(activation_data['mean_activations'])),
                    "activation_std": float(np.std(activation_data['mean_activations'])),
                    "top_feature_id": activation_data['top_features'][0]['feature_id'] if activation_data['top_features'] else None
                })
            
            if 'correlation_analysis' in results:
                correlation_data = results['correlation_analysis']
                layer_summary.update({
                    "num_high_correlations": correlation_data['num_high_correlations']
                })
            
            if 'importance_analysis' in results:
                importance_data = results['importance_analysis']
                layer_summary.update({
                    "mean_importance": float(np.mean(importance_data['importance_scores'])),
                    "top_important_feature_id": importance_data['top_important_features'][0]['feature_id'] if importance_data['top_important_features'] else None
                })
            
            layer_summaries[layer] = layer_summary
        
        summary["layer_summaries"] = layer_summaries
        
        # Save report
        report_file = output_path / 'feature_analysis_report.json'
        with open(report_file, 'w') as f:
            json.dump({
                "summary": summary,
                "detailed_results": analysis_results
            }, f, indent=2)
        
        # Save CSV summaries
        for layer, results in analysis_results.items():
            if 'activation_analysis' in results:
                activation_df = pd.DataFrame(results['activation_analysis']['top_features'])
                activation_df.to_csv(output_path / f'layer_{layer}_top_features.csv', index=False)
            
            if 'importance_analysis' in results:
                importance_df = pd.DataFrame(results['importance_analysis']['top_important_features'])
                importance_df.to_csv(output_path / f'layer_{layer}_important_features.csv', index=False)
        
        print("Analysis report generated successfully")
    
    def run_full_analysis(self, input_data: List[str], layers: List[int], output_dir: str) -> Dict:
        """
        Run full feature analysis pipeline.
        
        Args:
            input_data: List of input texts
            layers: List of layers to analyze
            output_dir: Directory to save results
            
        Returns:
            Complete analysis results
        """
        print("Starting full feature analysis pipeline")
        
        # Load model
        model = self.load_model()
        
        # Run analysis for each layer
        all_results = {}
        for layer in layers:
            print(f"\nAnalyzing layer {layer}...")
            
            # Extract features
            features = self.extract_features(model, input_data, layer)
            
            # Analyze activations
            activation_analysis = self.analyze_feature_activations(features)
            
            # Analyze correlations
            correlation_analysis = self.analyze_feature_correlations(features)
            
            # Analyze importance
            importance_analysis = self.analyze_feature_importance(features)
            
            # Combine results
            layer_results = {
                "features": features,
                "activation_analysis": activation_analysis,
                "correlation_analysis": correlation_analysis,
                "importance_analysis": importance_analysis
            }
            
            all_results[layer] = layer_results
        
        # Create visualizations
        self.create_feature_visualizations(all_results, output_dir)
        
        # Generate report
        self.generate_analysis_report(all_results, output_dir)
        
        print("Full feature analysis pipeline completed")
        return all_results

def main():
    """Main function for generic feature analysis."""
    parser = argparse.ArgumentParser(description="Generic feature analysis for AutoInterp")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--data", required=True, help="Path to input data file")
    parser.add_argument("--layers", nargs='+', type=int, default=[4, 10, 16, 22, 28], help="Layers to analyze")
    parser.add_argument("--output", default="feature_analysis_results", help="Output directory")
    parser.add_argument("--device", default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Load input data
    with open(args.data, 'r') as f:
        input_data = [line.strip() for line in f.readlines()]
    
    # Run analysis
    analyzer = GenericFeatureAnalysis(args.model, args.device)
    results = analyzer.run_full_analysis(input_data, args.layers, args.output)
    
    print(f"Analysis completed! Results saved to {args.output}")

if __name__ == "__main__":
    main()
