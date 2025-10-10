"""
Simple Llama-2-7b-hf SAE Trading Analysis
Uses existing feature mappings from results_summary_all_layers_large_model_yahoo_finance.csv
"""

import pandas as pd
import numpy as np
import torch
import os
import logging
from typing import Dict, List, Optional
from safetensors import safe_open

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaSAETrading:
    """Simple trading analyzer using Llama-2-7b-hf SAE features"""
    
    def __init__(self, 
                 model_path: str = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun",
                 layer: int = 16,
                 device: str = "cuda:0"):
        """Initialize the analyzer"""
        self.model_path = model_path
        self.layer = layer
        self.device = device
        
        # Load the SAE model
        self.sae_model = self._load_sae_model()
        
        # Load feature mappings from CSV
        self.feature_mappings = self._load_feature_mappings()
        
    def _load_sae_model(self):
        """Load the SAE model from the specified path"""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            # Load the base model
            model_name = "meta-llama/Llama-2-7b-hf"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16, device_map=self.device)
            
            # Load SAE weights
            sae_path = os.path.join(self.model_path, f"layers.{self.layer}")
            sae_weights_path = os.path.join(sae_path, "sae.safetensors")
            
            if os.path.exists(sae_weights_path):
                with safe_open(sae_weights_path, framework="pt") as f:
                    w_dec = f.get_tensor("W_dec").to(self.device)
                    w_enc = f.get_tensor("W_enc").to(self.device)
                    b_dec = f.get_tensor("b_dec").to(self.device)
                    b_enc = f.get_tensor("b_enc").to(self.device)
                
                logger.info(f"Loaded SAE weights from {sae_weights_path}")
                return {
                    'w_dec': w_dec,
                    'w_enc': w_enc,
                    'b_dec': b_dec,
                    'b_enc': b_enc,
                    'model': model,
                    'tokenizer': tokenizer
                }
            else:
                logger.error(f"SAE weights not found at {sae_weights_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading SAE model: {str(e)}")
            return None
    
    def _load_feature_mappings(self):
        """Load feature mappings from CSV file"""
        csv_path = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/FinanceLabeling/all_layers_financial_results/results_summary_all_layers_large_model_yahoo_finance.csv"
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Filter for layer 16 and get top features
            layer16_features = df[df['layer'] == 16].nlargest(20, 'f1_score')
            
            feature_mappings = {}
            for _, row in layer16_features.iterrows():
                feature_mappings[row['feature']] = {
                    'label': row['label'],
                    'f1_score': row['f1_score'],
                    'accuracy': row['accuracy']
                }
            
            logger.info(f"Loaded {len(feature_mappings)} features from layer 16")
            return feature_mappings
        else:
            logger.error(f"CSV file not found: {csv_path}")
            return {}
    
    def get_top_activations(self, text: str, top_n: int = 20) -> pd.DataFrame:
        """Get top SAE activations for a given text"""
        if self.sae_model is None:
            logger.error("SAE model not loaded")
            return pd.DataFrame()
        
        try:
            # Tokenize the input
            inputs = self.sae_model['tokenizer'](text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.sae_model['model'](**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[self.layer]
                
                # Encode through SAE
                h = torch.matmul(hidden_states, self.sae_model['w_enc']) + self.sae_model['b_enc']
                h_activated = torch.relu(h)
                
                # Get top activations
                activations = h_activated.abs().sum(dim=1)
                top_activations = activations.topk(top_n)
                
                # Create DataFrame
                df = pd.DataFrame({
                    'feature_id': top_activations.indices[0].cpu().numpy(),
                    'activation': top_activations.values[0].cpu().numpy()
                })
                
                # Add feature labels
                df['feature_label'] = df['feature_id'].apply(self._get_feature_label)
                
                return df
                
        except Exception as e:
            logger.error(f"Error getting activations: {str(e)}")
            return pd.DataFrame()
    
    def _get_feature_label(self, feature_id: int) -> str:
        """Get a human-readable label for a feature ID"""
        if feature_id in self.feature_mappings:
            return self.feature_mappings[feature_id]['label']
        return f"feature_{feature_id}"
    
    def analyze_headlines_for_features(self, headlines: List[str], top_n: int = 20) -> pd.DataFrame:
        """Analyze headlines and aggregate top activated features"""
        feature_sum_dict = {}
        feature_label_dict = {}
        
        for headline in headlines:
            logger.info(f"Processing headline: {headline[:50]}...")
            
            # Get top activations for the headline
            activations_df = self.get_top_activations(headline, top_n=top_n)
            
            if activations_df.empty:
                logger.warning(f"No activations found for headline: {headline}")
                continue
            
            # Accumulate activations
            for _, row in activations_df.iterrows():
                feature_id = row['feature_id']
                activation = row['activation']
                label = row['feature_label']
                
                if feature_id not in feature_sum_dict:
                    feature_sum_dict[feature_id] = activation
                    feature_label_dict[feature_id] = label
                else:
                    feature_sum_dict[feature_id] += activation
        
        # Convert to DataFrame
        aggregated_df = pd.DataFrame({
            'feature_id': list(feature_sum_dict.keys()),
            'total_activation': list(feature_sum_dict.values()),
            'feature_label': [feature_label_dict[f] for f in feature_sum_dict.keys()]
        })
        
        # Sort by total activation descending
        aggregated_df = aggregated_df.sort_values('total_activation', ascending=False)
        
        return aggregated_df
    
    def build_feature_matrix(self, headlines: List[str], returns: List[float], selected_features: List[int]) -> pd.DataFrame:
        """Build feature matrix for trading analysis"""
        if len(headlines) != len(returns):
            raise ValueError("Headlines and returns must have the same length")
        
        records = []
        
        for idx, (headline, ret) in enumerate(zip(headlines, returns)):
            logger.info(f"Processing headline {idx+1}/{len(headlines)}")
            
            # Get activations for this headline
            activations_df = self.get_top_activations(headline, top_n=50)
            
            if activations_df.empty:
                logger.warning(f"No activations found for headline: {headline}")
                continue
            
            # Build feature vector
            row_dict = {}
            
            # Add selected features
            for feature_id in selected_features:
                feature_name = f"feature_{feature_id}"
                if feature_id in activations_df['feature_id'].values:
                    row_dict[feature_name] = activations_df[activations_df['feature_id'] == feature_id]['activation'].iloc[0]
                else:
                    row_dict[feature_name] = 0.0
            
            # Add label (binary classification)
            row_dict['label'] = 1 if ret > 0.1 else 0
            row_dict['return'] = ret
            row_dict['headline'] = headline
            
            records.append(row_dict)
        
        return pd.DataFrame(records)

def main():
    """Main function to run the trading analysis"""
    logger.info("Starting Llama-2-7b-hf SAE Trading Analysis")
    
    # Initialize analyzer
    analyzer = LlamaSAETrading()
    
    if analyzer.sae_model is None:
        logger.error("Failed to load SAE model. Exiting.")
        return
    
    # Example financial headlines
    financial_headlines = [
        "Apple reports record quarterly earnings with strong iPhone sales",
        "Federal Reserve raises interest rates by 0.25%",
        "Tesla stock surges 15% after positive earnings report",
        "Market volatility increases due to geopolitical tensions",
        "Bitcoin reaches new all-time high above $100,000"
    ]
    
    # Example returns
    returns = [0.05, -0.02, 0.15, -0.08, 0.12]
    
    # Step 1: Analyze headlines for top features
    logger.info("Step 1: Analyzing headlines for top features...")
    top_features_df = analyzer.analyze_headlines_for_features(financial_headlines, top_n=20)
    
    logger.info(f"Found {len(top_features_df)} unique features")
    logger.info("Top 10 features by total activation:")
    for _, row in top_features_df.head(10).iterrows():
        logger.info(f"  {row['feature_label']}: {row['total_activation']:.4f}")
    
    # Step 2: Select top features for trading
    selected_features = top_features_df.head(10)['feature_id'].tolist()
    logger.info(f"Selected {len(selected_features)} features for trading analysis")
    
    # Step 3: Build feature matrix
    logger.info("Step 2: Building feature matrix...")
    feature_df = analyzer.build_feature_matrix(financial_headlines, returns, selected_features)
    
    logger.info(f"Created feature matrix with {len(feature_df)} samples and {len(feature_df.columns)} features")
    
    logger.info("Analysis completed successfully!")

if __name__ == "__main__":
    main()
