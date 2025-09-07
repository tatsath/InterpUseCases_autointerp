#!/usr/bin/env python3
"""
Generic Comparison Script for AutoInterp Analysis

This script provides generic comparison capabilities for analyzing
feature performance across different layers and analysis types.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
import os
from pathlib import Path
import argparse

class GenericComparison:
    """
    A class for generic comparison of AutoInterp analysis results.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the GenericComparison class.
        
        Args:
            data_dir: Directory containing analysis data
        """
        self.data_dir = Path(data_dir)
        self.comparison_data = {}
        self.statistics = {}
        
    def load_comparison_data(self) -> Dict:
        """
        Load data for comparison from various sources.
        
        Returns:
            Dictionary containing loaded data
        """
        print("Loading comparison data...")
        
        data = {}
        
        # Load from different result directories
        result_dirs = [
            "FinanceLabeling/multi_layer_full_results",
            "FinanceLabeling/multi_layer_lite_results",
            "FinanceLabeling/single_layer_full_results",
            "FinanceLabeling/single_layer_openrouter_results"
        ]
        
        for result_dir in result_dirs:
            dir_path = self.data_dir / result_dir
            if dir_path.exists():
                dir_data = self._load_directory_data(dir_path)
                if dir_data:
                    data[result_dir] = dir_data
        
        return data
    
    def _load_directory_data(self, dir_path: Path) -> Dict:
        """
        Load data from a specific directory.
        
        Args:
            dir_path: Path to the directory
            
        Returns:
            Dictionary containing loaded data
        """
        data = {}
        
        # Look for CSV files
        for csv_file in dir_path.glob("**/*.csv"):
            try:
                df = pd.read_csv(csv_file)
                data[csv_file.name] = df
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
        
        return data
    
    def compare_layers(self, data: Dict) -> pd.DataFrame:
        """
        Compare feature performance across different layers.
        
        Args:
            data: Dictionary containing analysis data
            
        Returns:
            DataFrame with layer comparison results
        """
        print("Comparing features across layers...")
        
        layer_comparisons = []
        
        for source, source_data in data.items():
            for filename, df in source_data.items():
                if 'feature' in filename.lower() and 'layer' in filename.lower():
                    # Extract layer information
                    layer_info = self._extract_layer_from_filename(filename)
                    if layer_info:
                        df['layer'] = layer_info
                        df['source'] = source
                        layer_comparisons.append(df)
        
        if layer_comparisons:
            combined_df = pd.concat(layer_comparisons, ignore_index=True)
            return combined_df
        else:
            return pd.DataFrame()
    
    def compare_analysis_types(self, data: Dict) -> pd.DataFrame:
        """
        Compare results across different analysis types.
        
        Args:
            data: Dictionary containing analysis data
            
        Returns:
            DataFrame with analysis type comparison results
        """
        print("Comparing analysis types...")
        
        analysis_comparisons = []
        
        for source, source_data in data.items():
            analysis_type = self._extract_analysis_type(source)
            for filename, df in source_data.items():
                if 'results_summary' in filename or 'features_' in filename:
                    df['analysis_type'] = analysis_type
                    df['source'] = source
                    analysis_comparisons.append(df)
        
        if analysis_comparisons:
            combined_df = pd.concat(analysis_comparisons, ignore_index=True)
            return combined_df
        else:
            return pd.DataFrame()
    
    def _extract_layer_from_filename(self, filename: str) -> Optional[int]:
        """
        Extract layer number from filename.
        
        Args:
            filename: Name of the file
            
        Returns:
            Layer number if found, None otherwise
        """
        import re
        
        # Look for patterns like "layer4", "layer_4", "layer10", etc.
        patterns = [
            r'layer(\d+)',
            r'layer_(\d+)',
            r'features_layer(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename.lower())
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        
        return None
    
    def _extract_analysis_type(self, source: str) -> str:
        """
        Extract analysis type from source path.
        
        Args:
            source: Source path string
            
        Returns:
            Analysis type string
        """
        if 'multi_layer_full' in source:
            return 'multi_layer_full'
        elif 'multi_layer_lite' in source:
            return 'multi_layer_lite'
        elif 'single_layer_full' in source:
            return 'single_layer_full'
        elif 'single_layer_openrouter' in source:
            return 'single_layer_openrouter'
        else:
            return 'unknown'
    
    def calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate comparison statistics.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary containing statistics
        """
        if df.empty:
            return {}
        
        stats = {}
        
        # Overall statistics
        if 'activation_score' in df.columns:
            stats['activation_score'] = {
                'mean': df['activation_score'].mean(),
                'std': df['activation_score'].std(),
                'min': df['activation_score'].min(),
                'max': df['activation_score'].max(),
                'median': df['activation_score'].median()
            }
        
        if 'interpretability_score' in df.columns:
            stats['interpretability_score'] = {
                'mean': df['interpretability_score'].mean(),
                'std': df['interpretability_score'].std(),
                'min': df['interpretability_score'].min(),
                'max': df['interpretability_score'].max(),
                'median': df['interpretability_score'].median()
            }
        
        if 'financial_relevance' in df.columns:
            stats['financial_relevance'] = {
                'mean': df['financial_relevance'].mean(),
                'std': df['financial_relevance'].std(),
                'min': df['financial_relevance'].min(),
                'max': df['financial_relevance'].max(),
                'median': df['financial_relevance'].median()
            }
        
        # Layer-wise statistics
        if 'layer' in df.columns:
            layer_stats = {}
            for layer in df['layer'].unique():
                layer_data = df[df['layer'] == layer]
                layer_stats[f'layer_{layer}'] = {
                    'count': len(layer_data),
                    'avg_activation': layer_data['activation_score'].mean() if 'activation_score' in layer_data.columns else None,
                    'avg_interpretability': layer_data['interpretability_score'].mean() if 'interpretability_score' in layer_data.columns else None,
                    'avg_financial_relevance': layer_data['financial_relevance'].mean() if 'financial_relevance' in layer_data.columns else None
                }
            stats['layer_wise'] = layer_stats
        
        # Analysis type statistics
        if 'analysis_type' in df.columns:
            type_stats = {}
            for analysis_type in df['analysis_type'].unique():
                type_data = df[df['analysis_type'] == analysis_type]
                type_stats[analysis_type] = {
                    'count': len(type_data),
                    'avg_activation': type_data['activation_score'].mean() if 'activation_score' in type_data.columns else None,
                    'avg_interpretability': type_data['interpretability_score'].mean() if 'interpretability_score' in type_data.columns else None,
                    'avg_financial_relevance': type_data['financial_relevance'].mean() if 'financial_relevance' in type_data.columns else None
                }
            stats['analysis_type_wise'] = type_stats
        
        return stats
    
    def create_comparison_visualizations(self, df: pd.DataFrame, output_dir: str):
        """
        Create visualizations for comparison analysis.
        
        Args:
            df: DataFrame to visualize
            output_dir: Directory to save visualizations
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"Creating comparison visualizations in {output_path}")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Layer comparison
        if 'layer' in df.columns and 'activation_score' in df.columns:
            plt.figure(figsize=(12, 8))
            
            # Box plot for activation scores by layer
            plt.subplot(2, 2, 1)
            df.boxplot(column='activation_score', by='layer', ax=plt.gca())
            plt.title('Activation Scores by Layer')
            plt.xlabel('Layer')
            plt.ylabel('Activation Score')
            
            # Line plot for average activation by layer
            plt.subplot(2, 2, 2)
            layer_avg = df.groupby('layer')['activation_score'].mean()
            layer_avg.plot(kind='line', marker='o')
            plt.title('Average Activation Score by Layer')
            plt.xlabel('Layer')
            plt.ylabel('Average Activation Score')
            plt.grid(True)
            
            # Interpretability scores by layer
            if 'interpretability_score' in df.columns:
                plt.subplot(2, 2, 3)
                df.boxplot(column='interpretability_score', by='layer', ax=plt.gca())
                plt.title('Interpretability Scores by Layer')
                plt.xlabel('Layer')
                plt.ylabel('Interpretability Score')
            
            # Financial relevance by layer
            if 'financial_relevance' in df.columns:
                plt.subplot(2, 2, 4)
                df.boxplot(column='financial_relevance', by='layer', ax=plt.gca())
                plt.title('Financial Relevance by Layer')
                plt.xlabel('Layer')
                plt.ylabel('Financial Relevance')
            
            plt.tight_layout()
            plt.savefig(output_path / 'layer_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Analysis type comparison
        if 'analysis_type' in df.columns:
            plt.figure(figsize=(12, 6))
            
            # Activation scores by analysis type
            plt.subplot(1, 2, 1)
            df.boxplot(column='activation_score', by='analysis_type', ax=plt.gca())
            plt.title('Activation Scores by Analysis Type')
            plt.xlabel('Analysis Type')
            plt.ylabel('Activation Score')
            plt.xticks(rotation=45)
            
            # Interpretability scores by analysis type
            if 'interpretability_score' in df.columns:
                plt.subplot(1, 2, 2)
                df.boxplot(column='interpretability_score', by='analysis_type', ax=plt.gca())
                plt.title('Interpretability Scores by Analysis Type')
                plt.xlabel('Analysis Type')
                plt.ylabel('Interpretability Score')
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_path / 'analysis_type_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Correlation heatmap
        if len(df.columns) > 2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                plt.figure(figsize=(10, 8))
                correlation_matrix = df[numeric_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
                plt.title('Feature Correlation Matrix')
                plt.tight_layout()
                plt.savefig(output_path / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    def generate_comparison_report(self, data: Dict, output_dir: str):
        """
        Generate a comprehensive comparison report.
        
        Args:
            data: Dictionary containing analysis data
            output_dir: Directory to save the report
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"Generating comparison report in {output_path}")
        
        # Load and compare data
        comparison_data = self.load_comparison_data()
        
        # Layer comparison
        layer_df = self.compare_layers(comparison_data)
        if not layer_df.empty:
            layer_df.to_csv(output_path / 'layer_comparison_data.csv', index=False)
            layer_stats = self.calculate_statistics(layer_df)
            self.create_comparison_visualizations(layer_df, output_path)
        
        # Analysis type comparison
        analysis_df = self.compare_analysis_types(comparison_data)
        if not analysis_df.empty:
            analysis_df.to_csv(output_path / 'analysis_type_comparison_data.csv', index=False)
            analysis_stats = self.calculate_statistics(analysis_df)
        
        # Generate summary report
        report = {
            "comparison_summary": {
                "total_sources": len(comparison_data),
                "layer_comparison_available": not layer_df.empty,
                "analysis_type_comparison_available": not analysis_df.empty
            },
            "statistics": {
                "layer_comparison": layer_stats if not layer_df.empty else {},
                "analysis_type_comparison": analysis_stats if not analysis_df.empty else {}
            },
            "recommendations": [
                "Use layer comparison to understand feature evolution",
                "Apply analysis type comparison to choose optimal analysis methods",
                "Monitor correlation patterns for feature relationships",
                "Leverage statistics for performance benchmarking"
            ]
        }
        
        with open(output_path / 'comparison_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("Comparison report generated successfully!")

def main():
    """Main function for generic comparison."""
    parser = argparse.ArgumentParser(description="Generic comparison of AutoInterp analysis results")
    parser.add_argument("--data_dir", default=".", help="Directory containing analysis data")
    parser.add_argument("--output_dir", default="comparison_results", help="Directory to save comparison results")
    
    args = parser.parse_args()
    
    comparator = GenericComparison(args.data_dir)
    comparator.generate_comparison_report({}, args.output_dir)

if __name__ == "__main__":
    main()
