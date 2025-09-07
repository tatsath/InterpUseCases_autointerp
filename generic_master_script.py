#!/usr/bin/env python3
"""
Generic Master Script for AutoInterp Analysis

This script orchestrates the complete AutoInterp analysis pipeline,
including feature extraction, labeling, comparison, and reporting.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Import the generic analysis modules
from generic_feature_analysis import GenericFeatureAnalysis
from generic_feature_labeling import GenericFeatureLabeling
from generic_comparison import GenericComparison
from generic_delphi_runner import GenericDelphiRunner
from consolidate_labels import LabelConsolidator

class AutoInterpMasterScript:
    """
    Master script for orchestrating AutoInterp analysis pipeline.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the AutoInterp master script.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path) if config_path else self._default_config()
        self.logger = self._setup_logging()
        self.results = {}
        
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
        Get default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "pipeline": {
                "stages": [
                    "feature_analysis",
                    "feature_labeling", 
                    "delphi_analysis",
                    "comparison",
                    "consolidation"
                ],
                "parallel_execution": False
            },
            "input": {
                "model_path": "path/to/model",
                "data_path": "path/to/data",
                "layers": [4, 10, 16, 22, 28]
            },
            "output": {
                "base_dir": "autointerp_results",
                "save_intermediate": True,
                "generate_report": True
            },
            "analysis": {
                "batch_size": 32,
                "max_tokens": 512,
                "device": "cuda"
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """
        Setup logging for the master script.
        
        Returns:
            Logger instance
        """
        logger = logging.getLogger('AutoInterpMasterScript')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler
            log_file = Path(self.config.get('output', {}).get('base_dir', 'autointerp_results')) / 'master_script.log'
            log_file.parent.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def run_feature_analysis(self, input_data: List[str], output_dir: str) -> Dict:
        """
        Run feature analysis stage.
        
        Args:
            input_data: List of input texts
            output_dir: Output directory
            
        Returns:
            Feature analysis results
        """
        self.logger.info("Starting feature analysis stage")
        
        analyzer = GenericFeatureAnalysis(
            self.config['input']['model_path'],
            self.config['analysis']['device']
        )
        
        results = analyzer.run_full_analysis(
            input_data,
            self.config['input']['layers'],
            output_dir
        )
        
        self.logger.info("Feature analysis stage completed")
        return results
    
    def run_feature_labeling(self, feature_data: Dict, output_dir: str) -> Dict:
        """
        Run feature labeling stage.
        
        Args:
            feature_data: Feature analysis results
            output_dir: Output directory
            
        Returns:
            Feature labeling results
        """
        self.logger.info("Starting feature labeling stage")
        
        labeler = GenericFeatureLabeling()
        results = labeler.run_labeling_pipeline(feature_data, output_dir)
        
        self.logger.info("Feature labeling stage completed")
        return results
    
    def run_delphi_analysis(self, feature_data: Dict, output_dir: str) -> Dict:
        """
        Run Delphi analysis stage.
        
        Args:
            feature_data: Feature analysis results
            output_dir: Output directory
            
        Returns:
            Delphi analysis results
        """
        self.logger.info("Starting Delphi analysis stage")
        
        delphi_runner = GenericDelphiRunner()
        results = delphi_runner.run_full_analysis(
            self.config['input']['data_path'],
            output_dir
        )
        
        self.logger.info("Delphi analysis stage completed")
        return results
    
    def run_comparison(self, output_dir: str) -> Dict:
        """
        Run comparison analysis stage.
        
        Args:
            output_dir: Output directory
            
        Returns:
            Comparison results
        """
        self.logger.info("Starting comparison analysis stage")
        
        comparator = GenericComparison(output_dir)
        results = comparator.generate_comparison_report({}, output_dir)
        
        self.logger.info("Comparison analysis stage completed")
        return results
    
    def run_consolidation(self, output_dir: str) -> Dict:
        """
        Run consolidation stage.
        
        Args:
            output_dir: Output directory
            
        Returns:
            Consolidation results
        """
        self.logger.info("Starting consolidation stage")
        
        consolidator = LabelConsolidator(output_dir)
        consolidator.save_consolidated_results(output_dir)
        
        self.logger.info("Consolidation stage completed")
        return {}
    
    def load_input_data(self) -> List[str]:
        """
        Load input data from file.
        
        Returns:
            List of input texts
        """
        data_path = self.config['input']['data_path']
        
        if not os.path.exists(data_path):
            self.logger.error(f"Input data file not found: {data_path}")
            return []
        
        try:
            with open(data_path, 'r') as f:
                data = [line.strip() for line in f.readlines() if line.strip()]
            
            self.logger.info(f"Loaded {len(data)} input samples")
            return data
        except Exception as e:
            self.logger.error(f"Error loading input data: {e}")
            return []
    
    def create_output_directory(self) -> str:
        """
        Create output directory with timestamp.
        
        Returns:
            Output directory path
        """
        base_dir = self.config['output']['base_dir']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(base_dir) / f"analysis_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Created output directory: {output_dir}")
        return str(output_dir)
    
    def save_pipeline_config(self, output_dir: str):
        """
        Save pipeline configuration to output directory.
        
        Args:
            output_dir: Output directory
        """
        config_file = Path(output_dir) / 'pipeline_config.json'
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        self.logger.info(f"Saved pipeline configuration to {config_file}")
    
    def generate_master_report(self, all_results: Dict, output_dir: str):
        """
        Generate a comprehensive master report.
        
        Args:
            all_results: Results from all pipeline stages
            output_dir: Output directory
        """
        self.logger.info("Generating master report")
        
        report = {
            "pipeline_summary": {
                "execution_time": datetime.now().isoformat(),
                "config_used": self.config,
                "stages_completed": list(all_results.keys()),
                "total_stages": len(self.config['pipeline']['stages'])
            },
            "results_summary": {
                "feature_analysis": "completed" if "feature_analysis" in all_results else "not_run",
                "feature_labeling": "completed" if "feature_labeling" in all_results else "not_run",
                "delphi_analysis": "completed" if "delphi_analysis" in all_results else "not_run",
                "comparison": "completed" if "comparison" in all_results else "not_run",
                "consolidation": "completed" if "consolidation" in all_results else "not_run"
            },
            "detailed_results": all_results
        }
        
        report_file = Path(output_dir) / 'master_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Master report saved to {report_file}")
    
    def run_pipeline(self, stages: List[str] = None) -> Dict:
        """
        Run the complete AutoInterp analysis pipeline.
        
        Args:
            stages: List of stages to run (optional)
            
        Returns:
            Complete pipeline results
        """
        if stages is None:
            stages = self.config['pipeline']['stages']
        
        self.logger.info(f"Starting AutoInterp pipeline with stages: {stages}")
        
        # Create output directory
        output_dir = self.create_output_directory()
        
        # Save configuration
        self.save_pipeline_config(output_dir)
        
        # Load input data
        input_data = self.load_input_data()
        if not input_data:
            self.logger.error("No input data loaded. Exiting pipeline.")
            return {}
        
        # Run pipeline stages
        all_results = {}
        
        for stage in stages:
            self.logger.info(f"Running pipeline stage: {stage}")
            
            try:
                if stage == "feature_analysis":
                    results = self.run_feature_analysis(input_data, output_dir)
                    all_results[stage] = results
                
                elif stage == "feature_labeling":
                    # Use feature analysis results if available
                    feature_data = all_results.get("feature_analysis", {})
                    if feature_data:
                        results = self.run_feature_labeling(feature_data, output_dir)
                        all_results[stage] = results
                    else:
                        self.logger.warning("Feature analysis results not available for labeling")
                
                elif stage == "delphi_analysis":
                    # Use feature analysis results if available
                    feature_data = all_results.get("feature_analysis", {})
                    if feature_data:
                        results = self.run_delphi_analysis(feature_data, output_dir)
                        all_results[stage] = results
                    else:
                        self.logger.warning("Feature analysis results not available for Delphi analysis")
                
                elif stage == "comparison":
                    results = self.run_comparison(output_dir)
                    all_results[stage] = results
                
                elif stage == "consolidation":
                    results = self.run_consolidation(output_dir)
                    all_results[stage] = results
                
                else:
                    self.logger.warning(f"Unknown pipeline stage: {stage}")
                
            except Exception as e:
                self.logger.error(f"Error in pipeline stage {stage}: {e}")
                continue
        
        # Generate master report
        if self.config['output']['generate_report']:
            self.generate_master_report(all_results, output_dir)
        
        self.logger.info("AutoInterp pipeline completed successfully")
        return all_results

def main():
    """Main function for the AutoInterp master script."""
    parser = argparse.ArgumentParser(description="AutoInterp Master Script")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--stages", nargs='+', help="Specific stages to run")
    parser.add_argument("--model", help="Path to model file")
    parser.add_argument("--data", help="Path to input data file")
    parser.add_argument("--output", help="Output directory")
    
    args = parser.parse_args()
    
    # Initialize master script
    master = AutoInterpMasterScript(args.config)
    
    # Override config with command line arguments
    if args.model:
        master.config['input']['model_path'] = args.model
    if args.data:
        master.config['input']['data_path'] = args.data
    if args.output:
        master.config['output']['base_dir'] = args.output
    
    # Run pipeline
    results = master.run_pipeline(args.stages)
    
    print("AutoInterp pipeline completed!")
    print(f"Results saved to: {master.config['output']['base_dir']}")

if __name__ == "__main__":
    main()
