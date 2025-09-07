#!/usr/bin/env python3
"""
Generic Delphi Runner for AutoInterp Analysis

This script provides a generic interface for running Delphi-based
interpretability analysis on different models and datasets.
"""

import torch
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import argparse
import logging
from datetime import datetime

class GenericDelphiRunner:
    """
    A generic runner for Delphi-based interpretability analysis.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the GenericDelphiRunner.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path) if config_path else self._default_config()
        self.results = {}
        self.logger = self._setup_logging()
        
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config from {config_path}: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """
        Get default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "model": {
                "name": "financial_language_model",
                "path": "path/to/model",
                "device": "cuda"
            },
            "analysis": {
                "layers": [4, 10, 16, 22, 28],
                "features_per_layer": 10,
                "batch_size": 32,
                "max_tokens": 512
            },
            "delphi": {
                "api_key": "your_api_key_here",
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 1000
            },
            "output": {
                "base_dir": "delphi_results",
                "save_explanations": True,
                "save_visualizations": True,
                "save_summaries": True
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """
        Setup logging for the runner.
        
        Returns:
            Logger instance
        """
        logger = logging.getLogger('GenericDelphiRunner')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_model(self) -> Any:
        """
        Load the model for analysis.
        
        Returns:
            Loaded model
        """
        self.logger.info(f"Loading model: {self.config['model']['name']}")
        
        # Placeholder for model loading logic
        # In a real implementation, this would load the actual model
        model = {
            "name": self.config['model']['name'],
            "path": self.config['model']['path'],
            "device": self.config['model']['device'],
            "loaded": True
        }
        
        self.logger.info("Model loaded successfully")
        return model
    
    def prepare_data(self, data_path: str) -> List[Dict]:
        """
        Prepare data for analysis.
        
        Args:
            data_path: Path to data file
            
        Returns:
            List of data samples
        """
        self.logger.info(f"Preparing data from {data_path}")
        
        # Placeholder for data preparation
        # In a real implementation, this would load and preprocess the data
        data = [
            {
                "id": 1,
                "text": "The company's quarterly earnings show strong growth with revenue increasing by 15% year-over-year.",
                "label": "positive"
            },
            {
                "id": 2,
                "text": "Market volatility remains high due to economic uncertainty and geopolitical tensions.",
                "label": "negative"
            }
        ]
        
        self.logger.info(f"Prepared {len(data)} data samples")
        return data
    
    def run_layer_analysis(self, model: Any, data: List[Dict], layer: int) -> Dict:
        """
        Run analysis for a specific layer.
        
        Args:
            model: Loaded model
            data: Data samples
            layer: Layer index
            
        Returns:
            Analysis results for the layer
        """
        self.logger.info(f"Running analysis for layer {layer}")
        
        # Placeholder for layer analysis
        # In a real implementation, this would run the actual analysis
        layer_results = {
            "layer": layer,
            "features_analyzed": self.config['analysis']['features_per_layer'],
            "data_samples": len(data),
            "timestamp": datetime.now().isoformat(),
            "features": []
        }
        
        # Generate placeholder feature results
        for i in range(self.config['analysis']['features_per_layer']):
            feature_result = {
                "feature_id": i + 1,
                "activation_score": np.random.uniform(0.7, 0.95),
                "interpretability_score": np.random.uniform(0.6, 0.9),
                "financial_relevance": np.random.uniform(0.7, 0.95),
                "interpretation": f"Feature {i + 1} analysis for layer {layer}"
            }
            layer_results["features"].append(feature_result)
        
        self.logger.info(f"Completed analysis for layer {layer}")
        return layer_results
    
    def run_delphi_analysis(self, layer_results: Dict) -> Dict:
        """
        Run Delphi analysis on layer results.
        
        Args:
            layer_results: Results from layer analysis
            
        Returns:
            Delphi analysis results
        """
        self.logger.info(f"Running Delphi analysis for layer {layer_results['layer']}")
        
        # Placeholder for Delphi analysis
        # In a real implementation, this would call the Delphi API
        delphi_results = {
            "layer": layer_results['layer'],
            "delphi_analysis": {
                "model_used": self.config['delphi']['model'],
                "temperature": self.config['delphi']['temperature'],
                "timestamp": datetime.now().isoformat(),
                "explanations": []
            }
        }
        
        # Generate placeholder explanations
        for feature in layer_results['features']:
            explanation = {
                "feature_id": feature['feature_id'],
                "explanation": f"Delphi explanation for feature {feature['feature_id']} in layer {layer_results['layer']}",
                "confidence": np.random.uniform(0.7, 0.95),
                "key_concepts": ["financial", "analysis", "interpretation"]
            }
            delphi_results['delphi_analysis']['explanations'].append(explanation)
        
        self.logger.info(f"Completed Delphi analysis for layer {layer_results['layer']}")
        return delphi_results
    
    def save_results(self, results: Dict, output_dir: str):
        """
        Save analysis results to files.
        
        Args:
            results: Analysis results
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        self.logger.info(f"Saving results to {output_path}")
        
        # Save detailed results
        if self.config['output']['save_explanations']:
            for layer, layer_results in results.items():
                layer_dir = output_path / f"layer_{layer}"
                layer_dir.mkdir(exist_ok=True)
                
                # Save explanations
                explanations_dir = layer_dir / "explanations"
                explanations_dir.mkdir(exist_ok=True)
                
                for feature in layer_results['features']:
                    explanation_file = explanations_dir / f"layers.{layer}_latent{feature['feature_id']}.txt"
                    with open(explanation_file, 'w') as f:
                        f.write(f"Feature Analysis: Layer {layer}, Latent {feature['feature_id']}\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(f"Activation Score: {feature['activation_score']:.3f}\n")
                        f.write(f"Interpretability Score: {feature['interpretability_score']:.3f}\n")
                        f.write(f"Financial Relevance: {feature['financial_relevance']:.3f}\n\n")
                        f.write(f"Interpretation: {feature['interpretation']}\n")
                
                # Save Delphi explanations
                if 'delphi_analysis' in layer_results:
                    for explanation in layer_results['delphi_analysis']['explanations']:
                        delphi_file = explanations_dir / f"delphi_layers.{layer}_latent{explanation['feature_id']}.txt"
                        with open(delphi_file, 'w') as f:
                            f.write(f"Delphi Analysis: Layer {layer}, Latent {explanation['feature_id']}\n")
                            f.write("=" * 50 + "\n\n")
                            f.write(f"Confidence: {explanation['confidence']:.3f}\n")
                            f.write(f"Key Concepts: {', '.join(explanation['key_concepts'])}\n\n")
                            f.write(f"Explanation: {explanation['explanation']}\n")
        
        # Save summary CSV
        if self.config['output']['save_summaries']:
            summary_data = []
            for layer, layer_results in results.items():
                for feature in layer_results['features']:
                    summary_data.append({
                        'layer': layer,
                        'feature_id': feature['feature_id'],
                        'activation_score': feature['activation_score'],
                        'interpretability_score': feature['interpretability_score'],
                        'financial_relevance': feature['financial_relevance'],
                        'interpretation': feature['interpretation']
                    })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(output_path / 'results_summary.csv', index=False)
        
        # Save configuration
        config_file = output_path / 'run_config.json'
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        self.logger.info("Results saved successfully")
    
    def run_full_analysis(self, data_path: str, output_dir: str = None) -> Dict:
        """
        Run full analysis pipeline.
        
        Args:
            data_path: Path to data file
            output_dir: Output directory (optional)
            
        Returns:
            Complete analysis results
        """
        if output_dir is None:
            output_dir = self.config['output']['base_dir']
        
        self.logger.info("Starting full analysis pipeline")
        
        # Load model and data
        model = self.load_model()
        data = self.prepare_data(data_path)
        
        # Run analysis for each layer
        all_results = {}
        for layer in self.config['analysis']['layers']:
            # Run layer analysis
            layer_results = self.run_layer_analysis(model, data, layer)
            
            # Run Delphi analysis
            delphi_results = self.run_delphi_analysis(layer_results)
            
            # Combine results
            combined_results = {**layer_results, **delphi_results}
            all_results[layer] = combined_results
        
        # Save results
        self.save_results(all_results, output_dir)
        
        self.logger.info("Full analysis pipeline completed")
        return all_results

def main():
    """Main function for generic Delphi runner."""
    parser = argparse.ArgumentParser(description="Generic Delphi runner for AutoInterp analysis")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--data", required=True, help="Path to data file")
    parser.add_argument("--output", help="Output directory")
    
    args = parser.parse_args()
    
    runner = GenericDelphiRunner(args.config)
    results = runner.run_full_analysis(args.data, args.output)
    
    print("Analysis completed successfully!")
    print(f"Results saved to: {args.output or runner.config['output']['base_dir']}")

if __name__ == "__main__":
    main()
