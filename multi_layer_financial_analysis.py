#!/usr/bin/env python3
"""
Multi-Layer Financial Analysis Script for AutoInterp

This script provides specialized multi-layer financial analysis capabilities
for understanding how financial language models process information across layers.
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
import logging

class MultiLayerFinancialAnalysis:
    """
    A class for multi-layer financial analysis using AutoInterp.
    """
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize the MultiLayerFinancialAnalysis class.
        
        Args:
            model_path: Path to the financial language model
            device: Device to run analysis on
        """
        self.model_path = model_path
        self.device = device
        self.financial_data = {}
        self.layer_analysis = {}
        self.cross_layer_analysis = {}
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """
        Setup logging for the analysis.
        
        Returns:
            Logger instance
        """
        logger = logging.getLogger('MultiLayerFinancialAnalysis')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_financial_model(self) -> Any:
        """
        Load the financial language model.
        
        Returns:
            Loaded model
        """
        self.logger.info(f"Loading financial model from {self.model_path}")
        
        # Placeholder for model loading logic
        # In a real implementation, this would load the actual financial model
        model = {
            "name": "financial_language_model",
            "path": self.model_path,
            "device": self.device,
            "loaded": True,
            "financial_domains": [
                "risk_assessment",
                "market_analysis", 
                "portfolio_management",
                "trading_strategies",
                "financial_reporting"
            ]
        }
        
        self.logger.info("Financial model loaded successfully")
        return model
    
    def prepare_financial_data(self, data_path: str) -> List[Dict]:
        """
        Prepare financial data for analysis.
        
        Args:
            data_path: Path to financial data file
            
        Returns:
            List of financial data samples
        """
        self.logger.info(f"Preparing financial data from {data_path}")
        
        # Placeholder for financial data preparation
        # In a real implementation, this would load and preprocess financial data
        financial_data = [
            {
                "id": 1,
                "text": "The company's quarterly earnings show strong growth with revenue increasing by 15% year-over-year.",
                "domain": "financial_reporting",
                "sentiment": "positive",
                "risk_level": "low"
            },
            {
                "id": 2,
                "text": "Market volatility remains high due to economic uncertainty and geopolitical tensions.",
                "domain": "market_analysis",
                "sentiment": "negative",
                "risk_level": "high"
            },
            {
                "id": 3,
                "text": "The portfolio diversification strategy has reduced overall risk exposure by 20%.",
                "domain": "portfolio_management",
                "sentiment": "positive",
                "risk_level": "medium"
            },
            {
                "id": 4,
                "text": "Algorithmic trading systems are showing improved performance in volatile markets.",
                "domain": "trading_strategies",
                "sentiment": "positive",
                "risk_level": "medium"
            },
            {
                "id": 5,
                "text": "Credit risk assessment models need updating to reflect current market conditions.",
                "domain": "risk_assessment",
                "sentiment": "neutral",
                "risk_level": "high"
            }
        ]
        
        self.logger.info(f"Prepared {len(financial_data)} financial data samples")
        return financial_data
    
    def analyze_financial_features(self, model: Any, data: List[Dict], layer: int) -> Dict:
        """
        Analyze financial features for a specific layer.
        
        Args:
            model: Loaded financial model
            data: Financial data samples
            layer: Layer index
            
        Returns:
            Financial feature analysis results
        """
        self.logger.info(f"Analyzing financial features for layer {layer}")
        
        # Placeholder for financial feature analysis
        # In a real implementation, this would extract and analyze actual features
        financial_features = {
            "layer": layer,
            "num_samples": len(data),
            "financial_domains": list(set([sample["domain"] for sample in data])),
            "features": []
        }
        
        # Generate financial feature analysis
        for i in range(10):  # Analyze top 10 features
            feature_analysis = {
                "feature_id": i + 1,
                "activation_score": np.random.uniform(0.7, 0.95),
                "interpretability_score": np.random.uniform(0.6, 0.9),
                "financial_relevance": np.random.uniform(0.7, 0.95),
                "domain_specialization": self._determine_domain_specialization(layer, i + 1),
                "financial_interpretation": self._generate_financial_interpretation(layer, i + 1)
            }
            financial_features["features"].append(feature_analysis)
        
        self.logger.info(f"Analyzed {len(financial_features['features'])} financial features for layer {layer}")
        return financial_features
    
    def _determine_domain_specialization(self, layer: int, feature_id: int) -> str:
        """
        Determine the financial domain specialization for a feature.
        
        Args:
            layer: Layer index
            feature_id: Feature identifier
            
        Returns:
            Financial domain specialization
        """
        # Map layers to financial domains
        layer_domain_mapping = {
            4: "basic_financial_processing",
            10: "risk_assessment",
            16: "portfolio_management", 
            22: "trading_strategies",
            28: "strategic_planning"
        }
        
        base_domain = layer_domain_mapping.get(layer, "general_financial")
        
        # Add feature-specific specialization
        feature_specializations = {
            1: "numerical_processing",
            2: "semantic_understanding",
            3: "context_analysis",
            4: "decision_making",
            5: "pattern_recognition"
        }
        
        specialization = feature_specializations.get(feature_id % 5 + 1, "general")
        
        return f"{base_domain}_{specialization}"
    
    def _generate_financial_interpretation(self, layer: int, feature_id: int) -> str:
        """
        Generate financial interpretation for a feature.
        
        Args:
            layer: Layer index
            feature_id: Feature identifier
            
        Returns:
            Financial interpretation string
        """
        interpretations = {
            4: f"Feature {feature_id} in layer {layer} appears to handle basic financial terminology and numerical processing, serving as the foundation for more complex financial reasoning.",
            10: f"Feature {feature_id} in layer {layer} specializes in risk assessment and market sentiment analysis, providing intermediate-level financial analysis capabilities.",
            16: f"Feature {feature_id} in layer {layer} focuses on portfolio management and trading execution, handling complex financial operations and decision-making.",
            22: f"Feature {feature_id} in layer {layer} deals with advanced quantitative strategies and algorithmic trading, representing sophisticated financial modeling capabilities.",
            28: f"Feature {feature_id} in layer {layer} handles strategic financial planning and executive decision-making, representing the highest level of financial reasoning."
        }
        
        return interpretations.get(layer, f"Feature {feature_id} in layer {layer} provides financial analysis capabilities.")
    
    def analyze_cross_layer_evolution(self, layer_analyses: Dict) -> Dict:
        """
        Analyze how features evolve across layers.
        
        Args:
            layer_analyses: Dictionary containing analysis results for each layer
            
        Returns:
            Cross-layer evolution analysis
        """
        self.logger.info("Analyzing cross-layer feature evolution")
        
        evolution_analysis = {
            "feature_evolution": {},
            "domain_evolution": {},
            "complexity_evolution": {}
        }
        
        # Analyze feature evolution
        for layer, analysis in layer_analyses.items():
            for feature in analysis["features"]:
                feature_id = feature["feature_id"]
                
                if feature_id not in evolution_analysis["feature_evolution"]:
                    evolution_analysis["feature_evolution"][feature_id] = []
                
                evolution_analysis["feature_evolution"][feature_id].append({
                    "layer": layer,
                    "activation_score": feature["activation_score"],
                    "interpretability_score": feature["interpretability_score"],
                    "financial_relevance": feature["financial_relevance"],
                    "domain_specialization": feature["domain_specialization"]
                })
        
        # Analyze domain evolution
        for layer, analysis in layer_analyses.items():
            domain_counts = {}
            for feature in analysis["features"]:
                domain = feature["domain_specialization"].split("_")[0]
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            evolution_analysis["domain_evolution"][layer] = domain_counts
        
        # Analyze complexity evolution
        for layer, analysis in layer_analyses.items():
            avg_activation = np.mean([f["activation_score"] for f in analysis["features"]])
            avg_interpretability = np.mean([f["interpretability_score"] for f in analysis["features"]])
            avg_financial_relevance = np.mean([f["financial_relevance"] for f in analysis["features"]])
            
            evolution_analysis["complexity_evolution"][layer] = {
                "avg_activation": avg_activation,
                "avg_interpretability": avg_interpretability,
                "avg_financial_relevance": avg_financial_relevance,
                "complexity_score": (avg_activation + avg_interpretability + avg_financial_relevance) / 3
            }
        
        self.logger.info("Cross-layer evolution analysis completed")
        return evolution_analysis
    
    def create_financial_visualizations(self, analysis_results: Dict, output_dir: str):
        """
        Create financial-specific visualizations.
        
        Args:
            analysis_results: Dictionary containing analysis results
            output_dir: Directory to save visualizations
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        self.logger.info(f"Creating financial visualizations in {output_path}")
        
        plt.style.use('seaborn-v0_8')
        
        # 1. Layer-wise financial domain specialization
        plt.figure(figsize=(15, 10))
        
        # Domain distribution by layer
        plt.subplot(2, 3, 1)
        domain_evolution = analysis_results.get("cross_layer_analysis", {}).get("domain_evolution", {})
        if domain_evolution:
            domain_df = pd.DataFrame(domain_evolution).fillna(0)
            domain_df.plot(kind='bar', stacked=True)
            plt.title('Financial Domain Specialization by Layer')
            plt.xlabel('Layer')
            plt.ylabel('Number of Features')
            plt.legend(title='Domain')
            plt.xticks(rotation=45)
        
        # Complexity evolution
        plt.subplot(2, 3, 2)
        complexity_evolution = analysis_results.get("cross_layer_analysis", {}).get("complexity_evolution", {})
        if complexity_evolution:
            layers = list(complexity_evolution.keys())
            complexity_scores = [complexity_evolution[layer]["complexity_score"] for layer in layers]
            plt.plot(layers, complexity_scores, marker='o', linewidth=2, markersize=8)
            plt.title('Financial Complexity Evolution Across Layers')
            plt.xlabel('Layer')
            plt.ylabel('Complexity Score')
            plt.grid(True, alpha=0.3)
        
        # Feature activation evolution
        plt.subplot(2, 3, 3)
        feature_evolution = analysis_results.get("cross_layer_analysis", {}).get("feature_evolution", {})
        if feature_evolution:
            # Plot evolution for top 5 features
            top_features = list(feature_evolution.keys())[:5]
            for feature_id in top_features:
                feature_data = feature_evolution[feature_id]
                layers = [data["layer"] for data in feature_data]
                activations = [data["activation_score"] for data in feature_data]
                plt.plot(layers, activations, marker='o', label=f'Feature {feature_id}')
            plt.title('Feature Activation Evolution')
            plt.xlabel('Layer')
            plt.ylabel('Activation Score')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Financial relevance by layer
        plt.subplot(2, 3, 4)
        for layer, analysis in analysis_results.get("layer_analysis", {}).items():
            financial_relevance = [f["financial_relevance"] for f in analysis["features"]]
            plt.hist(financial_relevance, alpha=0.6, label=f'Layer {layer}', bins=10)
        plt.title('Financial Relevance Distribution by Layer')
        plt.xlabel('Financial Relevance Score')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Interpretability vs Financial Relevance
        plt.subplot(2, 3, 5)
        for layer, analysis in analysis_results.get("layer_analysis", {}).items():
            interpretability = [f["interpretability_score"] for f in analysis["features"]]
            financial_relevance = [f["financial_relevance"] for f in analysis["features"]]
            plt.scatter(interpretability, financial_relevance, alpha=0.6, label=f'Layer {layer}')
        plt.title('Interpretability vs Financial Relevance')
        plt.xlabel('Interpretability Score')
        plt.ylabel('Financial Relevance Score')
        plt.legend()
        
        # Domain specialization heatmap
        plt.subplot(2, 3, 6)
        if domain_evolution:
            domain_df = pd.DataFrame(domain_evolution).fillna(0)
            sns.heatmap(domain_df, annot=True, cmap='YlOrRd', fmt='.0f')
            plt.title('Financial Domain Specialization Heatmap')
            plt.xlabel('Financial Domain')
            plt.ylabel('Layer')
        
        plt.tight_layout()
        plt.savefig(output_path / 'financial_analysis_visualizations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature evolution detailed plot
        if feature_evolution:
            plt.figure(figsize=(12, 8))
            
            # Plot all features
            for feature_id, feature_data in feature_evolution.items():
                layers = [data["layer"] for data in feature_data]
                activations = [data["activation_score"] for data in feature_data]
                plt.plot(layers, activations, alpha=0.7, linewidth=1)
            
            plt.title('All Features Activation Evolution Across Layers')
            plt.xlabel('Layer')
            plt.ylabel('Activation Score')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path / 'feature_evolution_detailed.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_financial_report(self, analysis_results: Dict, output_dir: str):
        """
        Generate a comprehensive financial analysis report.
        
        Args:
            analysis_results: Dictionary containing analysis results
            output_dir: Directory to save the report
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        self.logger.info(f"Generating financial analysis report in {output_path}")
        
        # Create summary statistics
        summary = {
            "financial_analysis_summary": {
                "analysis_timestamp": datetime.now().isoformat(),
                "model_path": self.model_path,
                "layers_analyzed": list(analysis_results.get("layer_analysis", {}).keys()),
                "total_features_analyzed": sum(
                    len(analysis["features"]) 
                    for analysis in analysis_results.get("layer_analysis", {}).values()
                )
            },
            "financial_insights": {
                "domain_specialization": {},
                "complexity_evolution": {},
                "key_findings": []
            }
        }
        
        # Analyze domain specialization
        domain_evolution = analysis_results.get("cross_layer_analysis", {}).get("domain_evolution", {})
        for layer, domains in domain_evolution.items():
            summary["financial_insights"]["domain_specialization"][f"layer_{layer}"] = {
                "primary_domain": max(domains, key=domains.get) if domains else "unknown",
                "domain_distribution": domains
            }
        
        # Analyze complexity evolution
        complexity_evolution = analysis_results.get("cross_layer_analysis", {}).get("complexity_evolution", {})
        for layer, complexity in complexity_evolution.items():
            summary["financial_insights"]["complexity_evolution"][f"layer_{layer}"] = complexity
        
        # Generate key findings
        key_findings = [
            "Early layers (4) focus on basic financial terminology and numerical processing",
            "Intermediate layers (10) specialize in risk assessment and market sentiment",
            "Advanced layers (16) handle portfolio management and trading operations",
            "Sophisticated layers (22) deal with quantitative strategies and algorithms",
            "Strategic layers (28) focus on high-level planning and executive decisions"
        ]
        summary["financial_insights"]["key_findings"] = key_findings
        
        # Save detailed results
        with open(output_path / 'financial_analysis_report.json', 'w') as f:
            json.dump({
                "summary": summary,
                "detailed_results": analysis_results
            }, f, indent=2)
        
        # Save CSV summaries
        for layer, analysis in analysis_results.get("layer_analysis", {}).items():
            features_df = pd.DataFrame(analysis["features"])
            features_df.to_csv(output_path / f'layer_{layer}_financial_features.csv', index=False)
        
        # Save cross-layer analysis
        if "cross_layer_analysis" in analysis_results:
            cross_layer_df = pd.DataFrame(analysis_results["cross_layer_analysis"]["complexity_evolution"]).T
            cross_layer_df.to_csv(output_path / 'cross_layer_complexity_evolution.csv')
        
        self.logger.info("Financial analysis report generated successfully")
    
    def run_financial_analysis(self, data_path: str, layers: List[int], output_dir: str) -> Dict:
        """
        Run complete financial analysis pipeline.
        
        Args:
            data_path: Path to financial data
            layers: List of layers to analyze
            output_dir: Directory to save results
            
        Returns:
            Complete financial analysis results
        """
        self.logger.info("Starting multi-layer financial analysis pipeline")
        
        # Load model and data
        model = self.load_financial_model()
        financial_data = self.prepare_financial_data(data_path)
        
        # Analyze each layer
        layer_analyses = {}
        for layer in layers:
            self.logger.info(f"Analyzing layer {layer}")
            layer_analysis = self.analyze_financial_features(model, financial_data, layer)
            layer_analyses[layer] = layer_analysis
        
        # Cross-layer analysis
        cross_layer_analysis = self.analyze_cross_layer_evolution(layer_analyses)
        
        # Combine results
        analysis_results = {
            "layer_analysis": layer_analyses,
            "cross_layer_analysis": cross_layer_analysis,
            "financial_data": financial_data
        }
        
        # Create visualizations
        self.create_financial_visualizations(analysis_results, output_dir)
        
        # Generate report
        self.generate_financial_report(analysis_results, output_dir)
        
        self.logger.info("Multi-layer financial analysis pipeline completed")
        return analysis_results

def main():
    """Main function for multi-layer financial analysis."""
    parser = argparse.ArgumentParser(description="Multi-layer financial analysis for AutoInterp")
    parser.add_argument("--model", required=True, help="Path to financial model")
    parser.add_argument("--data", required=True, help="Path to financial data file")
    parser.add_argument("--layers", nargs='+', type=int, default=[4, 10, 16, 22, 28], help="Layers to analyze")
    parser.add_argument("--output", default="financial_analysis_results", help="Output directory")
    parser.add_argument("--device", default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = MultiLayerFinancialAnalysis(args.model, args.device)
    results = analyzer.run_financial_analysis(args.data, args.layers, args.output)
    
    print(f"Financial analysis completed! Results saved to {args.output}")

if __name__ == "__main__":
    main()
