#!/usr/bin/env python3
"""
Run Financial Analysis Script for AutoInterp

This script provides a convenient interface for running financial analysis
using the AutoInterp framework with predefined configurations.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Import the financial analysis module
from multi_layer_financial_analysis import MultiLayerFinancialAnalysis

class FinancialAnalysisRunner:
    """
    Runner class for financial analysis using AutoInterp.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the FinancialAnalysisRunner.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path) if config_path else self._default_config()
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
            self.logger.error(f"Error loading config from {config_path}: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """
        Get default configuration for financial analysis.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "model": {
                "path": "path/to/financial_model",
                "device": "cuda",
                "type": "financial_language_model"
            },
            "data": {
                "path": "path/to/financial_data.txt",
                "format": "text",
                "domains": [
                    "risk_assessment",
                    "market_analysis",
                    "portfolio_management",
                    "trading_strategies",
                    "financial_reporting"
                ]
            },
            "analysis": {
                "layers": [4, 10, 16, 22, 28],
                "features_per_layer": 10,
                "batch_size": 32,
                "max_tokens": 512
            },
            "output": {
                "base_dir": "financial_analysis_results",
                "save_visualizations": True,
                "save_reports": True,
                "save_csv": True
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """
        Setup logging for the runner.
        
        Returns:
            Logger instance
        """
        logger = logging.getLogger('FinancialAnalysisRunner')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler
            log_file = Path(self.config['output']['base_dir']) / 'financial_analysis.log'
            log_file.parent.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def validate_config(self) -> bool:
        """
        Validate the configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        self.logger.info("Validating configuration")
        
        # Check model path
        if not os.path.exists(self.config['model']['path']):
            self.logger.warning(f"Model path does not exist: {self.config['model']['path']}")
        
        # Check data path
        if not os.path.exists(self.config['data']['path']):
            self.logger.warning(f"Data path does not exist: {self.config['data']['path']}")
        
        # Check layers
        if not self.config['analysis']['layers']:
            self.logger.error("No layers specified for analysis")
            return False
        
        # Check output directory
        output_dir = Path(self.config['output']['base_dir'])
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created output directory: {output_dir}")
        
        self.logger.info("Configuration validation completed")
        return True
    
    def create_output_directory(self) -> str:
        """
        Create timestamped output directory.
        
        Returns:
            Output directory path
        """
        base_dir = self.config['output']['base_dir']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(base_dir) / f"financial_analysis_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Created output directory: {output_dir}")
        return str(output_dir)
    
    def save_config(self, output_dir: str):
        """
        Save configuration to output directory.
        
        Args:
            output_dir: Output directory
        """
        config_file = Path(output_dir) / 'analysis_config.json'
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        self.logger.info(f"Saved configuration to {config_file}")
    
    def run_analysis(self, output_dir: str = None) -> Dict:
        """
        Run the financial analysis.
        
        Args:
            output_dir: Output directory (optional)
            
        Returns:
            Analysis results
        """
        if output_dir is None:
            output_dir = self.create_output_directory()
        
        self.logger.info("Starting financial analysis")
        
        # Validate configuration
        if not self.validate_config():
            self.logger.error("Configuration validation failed")
            return {}
        
        # Save configuration
        self.save_config(output_dir)
        
        # Initialize analyzer
        analyzer = MultiLayerFinancialAnalysis(
            self.config['model']['path'],
            self.config['model']['device']
        )
        
        # Run analysis
        results = analyzer.run_financial_analysis(
            self.config['data']['path'],
            self.config['analysis']['layers'],
            output_dir
        )
        
        self.logger.info("Financial analysis completed successfully")
        return results
    
    def run_quick_analysis(self, output_dir: str = None) -> Dict:
        """
        Run a quick financial analysis with reduced scope.
        
        Args:
            output_dir: Output directory (optional)
            
        Returns:
            Analysis results
        """
        if output_dir is None:
            output_dir = self.create_output_directory()
        
        self.logger.info("Starting quick financial analysis")
        
        # Override config for quick analysis
        quick_config = self.config.copy()
        quick_config['analysis']['layers'] = [4, 10, 28]  # Fewer layers
        quick_config['analysis']['features_per_layer'] = 5  # Fewer features
        
        # Save quick config
        config_file = Path(output_dir) / 'quick_analysis_config.json'
        with open(config_file, 'w') as f:
            json.dump(quick_config, f, indent=2)
        
        # Initialize analyzer
        analyzer = MultiLayerFinancialAnalysis(
            quick_config['model']['path'],
            quick_config['model']['device']
        )
        
        # Run analysis
        results = analyzer.run_financial_analysis(
            quick_config['data']['path'],
            quick_config['analysis']['layers'],
            output_dir
        )
        
        self.logger.info("Quick financial analysis completed successfully")
        return results
    
    def run_comparative_analysis(self, output_dir: str = None) -> Dict:
        """
        Run comparative analysis across different configurations.
        
        Args:
            output_dir: Output directory (optional)
            
        Returns:
            Comparative analysis results
        """
        if output_dir is None:
            output_dir = self.create_output_directory()
        
        self.logger.info("Starting comparative financial analysis")
        
        # Define different analysis configurations
        configs = {
            "early_layers": {"layers": [4, 10], "name": "Early Layers Analysis"},
            "middle_layers": {"layers": [10, 16, 22], "name": "Middle Layers Analysis"},
            "late_layers": {"layers": [22, 28], "name": "Late Layers Analysis"},
            "full_analysis": {"layers": [4, 10, 16, 22, 28], "name": "Full Analysis"}
        }
        
        comparative_results = {}
        
        for config_name, config in configs.items():
            self.logger.info(f"Running {config['name']}")
            
            # Create subdirectory for this configuration
            config_output_dir = Path(output_dir) / config_name
            config_output_dir.mkdir(exist_ok=True)
            
            # Initialize analyzer
            analyzer = MultiLayerFinancialAnalysis(
                self.config['model']['path'],
                self.config['model']['device']
            )
            
            # Run analysis
            results = analyzer.run_financial_analysis(
                self.config['data']['path'],
                config['layers'],
                str(config_output_dir)
            )
            
            comparative_results[config_name] = {
                "config": config,
                "results": results
            }
        
        # Generate comparative report
        self._generate_comparative_report(comparative_results, output_dir)
        
        self.logger.info("Comparative financial analysis completed successfully")
        return comparative_results
    
    def _generate_comparative_report(self, comparative_results: Dict, output_dir: str):
        """
        Generate a comparative analysis report.
        
        Args:
            comparative_results: Results from comparative analysis
            output_dir: Output directory
        """
        self.logger.info("Generating comparative analysis report")
        
        report = {
            "comparative_analysis_summary": {
                "analysis_timestamp": datetime.now().isoformat(),
                "configurations_compared": list(comparative_results.keys()),
                "total_configurations": len(comparative_results)
            },
            "configuration_results": {}
        }
        
        for config_name, config_results in comparative_results.items():
            config = config_results["config"]
            results = config_results["results"]
            
            # Extract summary statistics
            layer_analysis = results.get("layer_analysis", {})
            total_features = sum(len(analysis["features"]) for analysis in layer_analysis.values())
            
            report["configuration_results"][config_name] = {
                "name": config["name"],
                "layers_analyzed": config["layers"],
                "total_features": total_features,
                "layers_count": len(layer_analysis)
            }
        
        # Save comparative report
        report_file = Path(output_dir) / 'comparative_analysis_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Comparative report saved to {report_file}")

def main():
    """Main function for financial analysis runner."""
    parser = argparse.ArgumentParser(description="Financial Analysis Runner for AutoInterp")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--model", help="Path to financial model")
    parser.add_argument("--data", help="Path to financial data file")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--mode", choices=["full", "quick", "comparative"], default="full", 
                       help="Analysis mode")
    parser.add_argument("--layers", nargs='+', type=int, help="Layers to analyze")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = FinancialAnalysisRunner(args.config)
    
    # Override config with command line arguments
    if args.model:
        runner.config['model']['path'] = args.model
    if args.data:
        runner.config['data']['path'] = args.data
    if args.output:
        runner.config['output']['base_dir'] = args.output
    if args.layers:
        runner.config['analysis']['layers'] = args.layers
    
    # Run analysis based on mode
    if args.mode == "full":
        results = runner.run_analysis()
    elif args.mode == "quick":
        results = runner.run_quick_analysis()
    elif args.mode == "comparative":
        results = runner.run_comparative_analysis()
    
    print(f"Financial analysis completed in {args.mode} mode!")
    print(f"Results saved to: {runner.config['output']['base_dir']}")

if __name__ == "__main__":
    main()
