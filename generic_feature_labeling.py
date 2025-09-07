#!/usr/bin/env python3
"""
Generic Feature Labeling Script for AutoInterp

This script provides generic feature labeling capabilities for
automatically generating human-readable labels for neural network features.
"""

import torch
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import argparse
from datetime import datetime
import logging

class GenericFeatureLabeling:
    """
    A class for generic feature labeling and interpretation.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the GenericFeatureLabeling class.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path) if config_path else self._default_config()
        self.labeling_results = {}
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
            "labeling": {
                "method": "activation_analysis",
                "top_k_features": 50,
                "label_categories": [
                    "numerical_processing",
                    "semantic_understanding",
                    "context_analysis",
                    "decision_making",
                    "pattern_recognition"
                ]
            },
            "output": {
                "save_detailed_labels": True,
                "save_summary": True,
                "save_visualizations": True
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """
        Setup logging for the labeler.
        
        Returns:
            Logger instance
        """
        logger = logging.getLogger('GenericFeatureLabeling')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def analyze_feature_activations(self, feature_data: Dict) -> List[Dict]:
        """
        Analyze feature activations to identify patterns.
        
        Args:
            feature_data: Dictionary containing feature data
            
        Returns:
            List of feature analysis results
        """
        self.logger.info("Analyzing feature activations for labeling")
        
        features = feature_data.get('features', [])
        layer = feature_data.get('layer', 'unknown')
        
        labeled_features = []
        
        for feature in features:
            feature_id = feature.get('feature_id', 0)
            activation_score = feature.get('activation_score', 0.0)
            interpretability_score = feature.get('interpretability_score', 0.0)
            financial_relevance = feature.get('financial_relevance', 0.0)
            
            # Determine feature category based on scores
            category = self._determine_feature_category(
                activation_score, interpretability_score, financial_relevance
            )
            
            # Generate label based on category and scores
            label = self._generate_feature_label(feature_id, category, activation_score)
            
            # Create detailed interpretation
            interpretation = self._generate_interpretation(
                feature_id, category, activation_score, interpretability_score, financial_relevance
            )
            
            labeled_feature = {
                "feature_id": feature_id,
                "layer": layer,
                "category": category,
                "label": label,
                "activation_score": activation_score,
                "interpretability_score": interpretability_score,
                "financial_relevance": financial_relevance,
                "interpretation": interpretation,
                "confidence": self._calculate_label_confidence(activation_score, interpretability_score)
            }
            
            labeled_features.append(labeled_feature)
        
        self.logger.info(f"Labeled {len(labeled_features)} features for layer {layer}")
        return labeled_features
    
    def _determine_feature_category(self, activation: float, interpretability: float, financial_relevance: float) -> str:
        """
        Determine the category of a feature based on its scores.
        
        Args:
            activation: Activation score
            interpretability: Interpretability score
            financial_relevance: Financial relevance score
            
        Returns:
            Feature category string
        """
        # High activation and high financial relevance -> decision making
        if activation > 0.9 and financial_relevance > 0.9:
            return "decision_making"
        
        # High interpretability and moderate activation -> semantic understanding
        elif interpretability > 0.8 and 0.7 < activation < 0.9:
            return "semantic_understanding"
        
        # High activation and low interpretability -> numerical processing
        elif activation > 0.8 and interpretability < 0.7:
            return "numerical_processing"
        
        # Moderate scores across all -> context analysis
        elif 0.6 < activation < 0.8 and 0.6 < interpretability < 0.8:
            return "context_analysis"
        
        # Low scores -> pattern recognition
        else:
            return "pattern_recognition"
    
    def _generate_feature_label(self, feature_id: int, category: str, activation: float) -> str:
        """
        Generate a human-readable label for a feature.
        
        Args:
            feature_id: Feature identifier
            category: Feature category
            activation: Activation score
            
        Returns:
            Human-readable label
        """
        base_labels = {
            "decision_making": [
                "Investment Decision Circuit",
                "Risk Assessment Hub",
                "Strategic Planning Node",
                "Financial Decision Center",
                "Portfolio Optimization Circuit"
            ],
            "semantic_understanding": [
                "Financial Concept Processor",
                "Market Sentiment Analyzer",
                "Context Understanding Node",
                "Semantic Integration Hub",
                "Meaning Construction Circuit"
            ],
            "numerical_processing": [
                "Numerical Data Processor",
                "Mathematical Operation Circuit",
                "Calculation Engine",
                "Statistical Analysis Node",
                "Quantitative Processing Hub"
            ],
            "context_analysis": [
                "Context Integration Circuit",
                "Situational Analysis Node",
                "Environmental Processor",
                "Contextual Understanding Hub",
                "Situational Awareness Circuit"
            ],
            "pattern_recognition": [
                "Pattern Detection Circuit",
                "Feature Recognition Node",
                "Pattern Analysis Hub",
                "Recognition Engine",
                "Pattern Matching Circuit"
            ]
        }
        
        # Select label based on activation score
        labels = base_labels.get(category, ["Generic Feature"])
        label_index = int((activation - 0.5) * len(labels)) % len(labels)
        
        return labels[label_index]
    
    def _generate_interpretation(self, feature_id: int, category: str, activation: float, 
                               interpretability: float, financial_relevance: float) -> str:
        """
        Generate a detailed interpretation for a feature.
        
        Args:
            feature_id: Feature identifier
            category: Feature category
            activation: Activation score
            interpretability: Interpretability score
            financial_relevance: Financial relevance score
            
        Returns:
            Detailed interpretation string
        """
        interpretations = {
            "decision_making": f"Feature {feature_id} appears to be a high-level decision-making circuit with strong activation ({activation:.3f}) and high financial relevance ({financial_relevance:.3f}). This feature likely plays a crucial role in final investment decisions and strategic planning.",
            
            "semantic_understanding": f"Feature {feature_id} demonstrates strong semantic understanding capabilities with high interpretability ({interpretability:.3f}) and moderate activation ({activation:.3f}). This feature likely processes financial concepts and market sentiment information.",
            
            "numerical_processing": f"Feature {feature_id} shows strong numerical processing capabilities with high activation ({activation:.3f}) but lower interpretability ({interpretability:.3f}). This feature likely handles mathematical operations and quantitative financial data.",
            
            "context_analysis": f"Feature {feature_id} exhibits balanced performance across activation ({activation:.3f}) and interpretability ({interpretability:.3f}) metrics. This feature likely processes contextual information and situational awareness in financial scenarios.",
            
            "pattern_recognition": f"Feature {feature_id} shows lower activation ({activation:.3f}) and interpretability ({interpretability:.3f}) scores, suggesting it may be involved in pattern recognition and feature detection tasks."
        }
        
        return interpretations.get(category, f"Feature {feature_id} analysis: activation={activation:.3f}, interpretability={interpretability:.3f}, financial_relevance={financial_relevance:.3f}")
    
    def _calculate_label_confidence(self, activation: float, interpretability: float) -> float:
        """
        Calculate confidence in the generated label.
        
        Args:
            activation: Activation score
            interpretability: Interpretability score
            
        Returns:
            Confidence score between 0 and 1
        """
        # Higher confidence for features with both high activation and interpretability
        confidence = (activation + interpretability) / 2
        
        # Boost confidence for very high scores
        if activation > 0.9 and interpretability > 0.8:
            confidence = min(1.0, confidence + 0.1)
        
        return confidence
    
    def consolidate_labels(self, all_labeled_features: Dict) -> pd.DataFrame:
        """
        Consolidate labels from multiple layers.
        
        Args:
            all_labeled_features: Dictionary containing labeled features from all layers
            
        Returns:
            Consolidated DataFrame
        """
        self.logger.info("Consolidating labels from all layers")
        
        all_features = []
        for layer, features in all_labeled_features.items():
            for feature in features:
                feature['layer'] = layer
                all_features.append(feature)
        
        consolidated_df = pd.DataFrame(all_features)
        
        # Add cross-layer analysis
        if not consolidated_df.empty:
            # Group by feature_id to find features that appear in multiple layers
            feature_counts = consolidated_df['feature_id'].value_counts()
            multi_layer_features = feature_counts[feature_counts > 1].index.tolist()
            
            # Add multi-layer flag
            consolidated_df['appears_in_multiple_layers'] = consolidated_df['feature_id'].isin(multi_layer_features)
            
            # Calculate average scores across layers
            avg_scores = consolidated_df.groupby('feature_id').agg({
                'activation_score': 'mean',
                'interpretability_score': 'mean',
                'financial_relevance': 'mean',
                'confidence': 'mean'
            }).reset_index()
            
            avg_scores.columns = ['feature_id', 'avg_activation_score', 'avg_interpretability_score', 
                                'avg_financial_relevance', 'avg_confidence']
            
            consolidated_df = consolidated_df.merge(avg_scores, on='feature_id', how='left')
        
        self.logger.info(f"Consolidated {len(consolidated_df)} feature labels")
        return consolidated_df
    
    def generate_labeling_report(self, consolidated_df: pd.DataFrame, output_dir: str):
        """
        Generate a comprehensive labeling report.
        
        Args:
            consolidated_df: Consolidated DataFrame with all labels
            output_dir: Directory to save the report
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        self.logger.info(f"Generating labeling report in {output_path}")
        
        # Save detailed labels
        if self.config['output']['save_detailed_labels']:
            consolidated_df.to_csv(output_path / 'detailed_feature_labels.csv', index=False)
            
            # Save by category
            for category in consolidated_df['category'].unique():
                category_df = consolidated_df[consolidated_df['category'] == category]
                category_df.to_csv(output_path / f'labels_{category}.csv', index=False)
        
        # Generate summary statistics
        if self.config['output']['save_summary']:
            summary_stats = {
                "labeling_summary": {
                    "total_features_labeled": len(consolidated_df),
                    "unique_features": consolidated_df['feature_id'].nunique(),
                    "layers_analyzed": sorted(consolidated_df['layer'].unique().tolist()),
                    "categories_identified": consolidated_df['category'].value_counts().to_dict()
                },
                "quality_metrics": {
                    "average_confidence": consolidated_df['confidence'].mean(),
                    "high_confidence_features": len(consolidated_df[consolidated_df['confidence'] > 0.8]),
                    "multi_layer_features": consolidated_df['appears_in_multiple_layers'].sum()
                },
                "category_analysis": {}
            }
            
            # Category-specific analysis
            for category in consolidated_df['category'].unique():
                category_data = consolidated_df[consolidated_df['category'] == category]
                summary_stats["category_analysis"][category] = {
                    "count": len(category_data),
                    "avg_activation": category_data['activation_score'].mean(),
                    "avg_interpretability": category_data['interpretability_score'].mean(),
                    "avg_financial_relevance": category_data['financial_relevance'].mean(),
                    "avg_confidence": category_data['confidence'].mean()
                }
            
            # Save summary
            with open(output_path / 'labeling_summary.json', 'w') as f:
                json.dump(summary_stats, f, indent=2)
        
        # Create visualizations
        if self.config['output']['save_visualizations']:
            self._create_labeling_visualizations(consolidated_df, output_path)
        
        self.logger.info("Labeling report generated successfully")
    
    def _create_labeling_visualizations(self, df: pd.DataFrame, output_path: Path):
        """
        Create visualizations for labeling results.
        
        Args:
            df: DataFrame with labeling results
            output_path: Path to save visualizations
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.style.use('seaborn-v0_8')
        
        # 1. Category distribution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        category_counts = df['category'].value_counts()
        plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
        plt.title('Feature Category Distribution')
        
        # 2. Confidence distribution
        plt.subplot(2, 2, 2)
        plt.hist(df['confidence'], bins=30, alpha=0.7)
        plt.title('Label Confidence Distribution')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        
        # 3. Activation vs Interpretability by category
        plt.subplot(2, 2, 3)
        for category in df['category'].unique():
            category_data = df[df['category'] == category]
            plt.scatter(category_data['activation_score'], category_data['interpretability_score'], 
                       label=category, alpha=0.6)
        plt.xlabel('Activation Score')
        plt.ylabel('Interpretability Score')
        plt.title('Activation vs Interpretability by Category')
        plt.legend()
        
        # 4. Layer-wise category distribution
        plt.subplot(2, 2, 4)
        layer_category = pd.crosstab(df['layer'], df['category'])
        layer_category.plot(kind='bar', stacked=True)
        plt.title('Category Distribution by Layer')
        plt.xlabel('Layer')
        plt.ylabel('Number of Features')
        plt.legend(title='Category')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / 'labeling_visualizations.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_labeling_pipeline(self, feature_data: Dict, output_dir: str) -> pd.DataFrame:
        """
        Run the complete feature labeling pipeline.
        
        Args:
            feature_data: Dictionary containing feature data from all layers
            output_dir: Directory to save results
            
        Returns:
            Consolidated DataFrame with all labels
        """
        self.logger.info("Starting feature labeling pipeline")
        
        # Label features for each layer
        all_labeled_features = {}
        for layer, layer_data in feature_data.items():
            self.logger.info(f"Labeling features for layer {layer}")
            labeled_features = self.analyze_feature_activations(layer_data)
            all_labeled_features[layer] = labeled_features
        
        # Consolidate labels
        consolidated_df = self.consolidate_labels(all_labeled_features)
        
        # Generate report
        self.generate_labeling_report(consolidated_df, output_dir)
        
        self.logger.info("Feature labeling pipeline completed")
        return consolidated_df

def main():
    """Main function for generic feature labeling."""
    parser = argparse.ArgumentParser(description="Generic feature labeling for AutoInterp")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--input", required=True, help="Path to input feature data file")
    parser.add_argument("--output", default="labeling_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Load input data
    with open(args.input, 'r') as f:
        feature_data = json.load(f)
    
    # Run labeling
    labeler = GenericFeatureLabeling(args.config)
    results = labeler.run_labeling_pipeline(feature_data, args.output)
    
    print(f"Labeling completed! Results saved to {args.output}")
    print(f"Total features labeled: {len(results)}")

if __name__ == "__main__":
    main()
