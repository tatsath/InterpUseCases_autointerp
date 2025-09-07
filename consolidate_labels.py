#!/usr/bin/env python3
"""
Consolidate Labels Script for AutoInterp Analysis

This script consolidates feature labels from multiple analysis runs
and creates a unified dataset for further analysis.
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

class LabelConsolidator:
    """
    A class for consolidating feature labels from multiple analysis runs.
    """
    
    def __init__(self, results_dir: str):
        """
        Initialize the LabelConsolidator.
        
        Args:
            results_dir: Directory containing analysis results
        """
        self.results_dir = Path(results_dir)
        self.consolidated_data = {}
        self.feature_labels = {}
        
    def load_results_from_directory(self, subdir: str) -> Dict:
        """
        Load results from a specific subdirectory.
        
        Args:
            subdir: Subdirectory name (e.g., 'multi_layer_full_results')
            
        Returns:
            Dictionary containing loaded results
        """
        subdir_path = self.results_dir / subdir
        if not subdir_path.exists():
            print(f"Warning: Directory {subdir_path} does not exist")
            return {}
        
        results = {}
        
        # Look for CSV files with results
        for csv_file in subdir_path.glob("**/*.csv"):
            try:
                df = pd.read_csv(csv_file)
                results[csv_file.name] = df
                print(f"Loaded {csv_file.name} with {len(df)} rows")
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
        
        return results
    
    def consolidate_multi_layer_results(self) -> pd.DataFrame:
        """
        Consolidate results from multi-layer analysis.
        
        Returns:
            Consolidated DataFrame
        """
        print("Consolidating multi-layer results...")
        
        all_results = []
        
        # Load from multi_layer_full_results
        full_results = self.load_results_from_directory("multi_layer_full_results")
        for filename, df in full_results.items():
            if "results_summary" in filename:
                # Extract layer information from directory structure
                layer_info = self._extract_layer_info(filename)
                df['layer'] = layer_info.get('layer', 'unknown')
                df['analysis_type'] = 'multi_layer_full'
                all_results.append(df)
        
        # Load from multi_layer_lite_results
        lite_results = self.load_results_from_directory("multi_layer_lite_results")
        for filename, df in lite_results.items():
            if "features_layer" in filename:
                layer_info = self._extract_layer_info(filename)
                df['layer'] = layer_info.get('layer', 'unknown')
                df['analysis_type'] = 'multi_layer_lite'
                all_results.append(df)
        
        if all_results:
            consolidated = pd.concat(all_results, ignore_index=True)
            return consolidated
        else:
            return pd.DataFrame()
    
    def consolidate_single_layer_results(self) -> pd.DataFrame:
        """
        Consolidate results from single-layer analysis.
        
        Returns:
            Consolidated DataFrame
        """
        print("Consolidating single-layer results...")
        
        all_results = []
        
        # Load from single_layer_full_results
        full_results = self.load_results_from_directory("single_layer_full_results")
        for filename, df in full_results.items():
            if "results_summary" in filename:
                layer_info = self._extract_layer_info(filename)
                df['layer'] = layer_info.get('layer', 'unknown')
                df['analysis_type'] = 'single_layer_full'
                all_results.append(df)
        
        # Load from single_layer_openrouter_results
        openrouter_results = self.load_results_from_directory("single_layer_openrouter_results")
        for filename, df in openrouter_results.items():
            if "results_summary" in filename:
                layer_info = self._extract_layer_info(filename)
                df['layer'] = layer_info.get('layer', 'unknown')
                df['analysis_type'] = 'single_layer_openrouter'
                all_results.append(df)
        
        if all_results:
            consolidated = pd.concat(all_results, ignore_index=True)
            return consolidated
        else:
            return pd.DataFrame()
    
    def _extract_layer_info(self, filename: str) -> Dict:
        """
        Extract layer information from filename.
        
        Args:
            filename: Name of the file
            
        Returns:
            Dictionary containing layer information
        """
        layer_info = {}
        
        # Extract layer number from filename
        if "layer" in filename.lower():
            parts = filename.lower().split("layer")
            if len(parts) > 1:
                layer_part = parts[1].split("_")[0]
                try:
                    layer_info['layer'] = int(layer_part)
                except ValueError:
                    layer_info['layer'] = 'unknown'
        
        return layer_info
    
    def create_unified_labels(self, consolidated_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create unified feature labels from consolidated data.
        
        Args:
            consolidated_df: Consolidated DataFrame
            
        Returns:
            DataFrame with unified labels
        """
        print("Creating unified feature labels...")
        
        if consolidated_df.empty:
            return pd.DataFrame()
        
        # Group by feature_id and create unified labels
        unified_labels = []
        
        for feature_id in consolidated_df['feature_id'].unique():
            feature_data = consolidated_df[consolidated_df['feature_id'] == feature_id]
            
            # Calculate average scores
            avg_activation = feature_data['activation_score'].mean()
            avg_interpretability = feature_data['interpretability_score'].mean()
            avg_financial_relevance = feature_data['financial_relevance'].mean()
            
            # Get most common interpretation
            interpretations = feature_data['interpretation'].dropna()
            if not interpretations.empty:
                most_common_interpretation = interpretations.mode().iloc[0] if not interpretations.mode().empty else interpretations.iloc[0]
            else:
                most_common_interpretation = "Unknown"
            
            # Get layer information
            layers = feature_data['layer'].unique()
            analysis_types = feature_data['analysis_type'].unique()
            
            unified_label = {
                'feature_id': feature_id,
                'avg_activation_score': avg_activation,
                'avg_interpretability_score': avg_interpretability,
                'avg_financial_relevance': avg_financial_relevance,
                'unified_interpretation': most_common_interpretation,
                'layers_analyzed': list(layers),
                'analysis_types': list(analysis_types),
                'num_analyses': len(feature_data)
            }
            
            unified_labels.append(unified_label)
        
        return pd.DataFrame(unified_labels)
    
    def save_consolidated_results(self, output_dir: str):
        """
        Save consolidated results to files.
        
        Args:
            output_dir: Directory to save consolidated results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"Saving consolidated results to {output_path}")
        
        # Consolidate multi-layer results
        multi_layer_df = self.consolidate_multi_layer_results()
        if not multi_layer_df.empty:
            multi_layer_df.to_csv(output_path / "consolidated_multi_layer_results.csv", index=False)
            print(f"Saved multi-layer results: {len(multi_layer_df)} rows")
        
        # Consolidate single-layer results
        single_layer_df = self.consolidate_single_layer_results()
        if not single_layer_df.empty:
            single_layer_df.to_csv(output_path / "consolidated_single_layer_results.csv", index=False)
            print(f"Saved single-layer results: {len(single_layer_df)} rows")
        
        # Create unified labels
        all_results = []
        if not multi_layer_df.empty:
            all_results.append(multi_layer_df)
        if not single_layer_df.empty:
            all_results.append(single_layer_df)
        
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            unified_labels = self.create_unified_labels(combined_df)
            
            if not unified_labels.empty:
                unified_labels.to_csv(output_path / "unified_feature_labels.csv", index=False)
                print(f"Saved unified labels: {len(unified_labels)} features")
                
                # Save summary statistics
                summary_stats = {
                    "total_features": len(unified_labels),
                    "avg_activation_score": unified_labels['avg_activation_score'].mean(),
                    "avg_interpretability_score": unified_labels['avg_interpretability_score'].mean(),
                    "avg_financial_relevance": unified_labels['avg_financial_relevance'].mean(),
                    "layers_analyzed": list(set([layer for layers in unified_labels['layers_analyzed'] for layer in layers])),
                    "analysis_types": list(set([atype for atypes in unified_labels['analysis_types'] for atype in atypes]))
                }
                
                with open(output_path / "consolidation_summary.json", 'w') as f:
                    json.dump(summary_stats, f, indent=2)
                
                print("Consolidation complete!")

def main():
    """Main function for label consolidation."""
    parser = argparse.ArgumentParser(description="Consolidate feature labels from AutoInterp analysis")
    parser.add_argument("--results_dir", default="FinanceLabeling", help="Directory containing analysis results")
    parser.add_argument("--output_dir", default="consolidated_results", help="Directory to save consolidated results")
    
    args = parser.parse_args()
    
    consolidator = LabelConsolidator(args.results_dir)
    consolidator.save_consolidated_results(args.output_dir)

if __name__ == "__main__":
    main()
