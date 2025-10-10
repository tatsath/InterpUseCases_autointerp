"""
Minimal Llama-2-7b-hf SAE Trading Analysis
Just the core functions from the notebook
"""

import pandas as pd
import numpy as np
import torch
import os
import logging
from typing import Dict, List
from safetensors import safe_open

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaSAEMinimal:
    """Minimal trading analyzer using Llama-2-7b-hf SAE features"""
    
    def __init__(self, layer: int = 16, device: str = "cuda:1"):
        self.layer = layer
        self.device = device
        self.sae_model = self._load_sae_model()
        self.feature_mappings = self._load_feature_mappings()
        
    def _load_sae_model(self):
        """Load the SAE model"""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            model_name = "meta-llama/Llama-2-7b-hf"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Set padding token for the tokenizer
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16, device_map=self.device, low_cpu_mem_usage=True)
            
            model_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
            sae_path = os.path.join(model_path, f"layers.{self.layer}")
            sae_weights_path = os.path.join(sae_path, "sae.safetensors")
            
            if os.path.exists(sae_weights_path):
                with safe_open(sae_weights_path, framework="pt") as f:
                    w_dec = f.get_tensor("W_dec").to(self.device)
                    w_enc = f.get_tensor("encoder.weight").to(self.device)
                    b_dec = f.get_tensor("b_dec").to(self.device)
                    b_enc = f.get_tensor("encoder.bias").to(self.device)
                
                return {
                    'w_dec': w_dec, 'w_enc': w_enc, 'b_dec': b_dec, 'b_enc': b_enc,
                    'model': model, 'tokenizer': tokenizer
                }
            return None
        except Exception as e:
            logger.error(f"Error loading SAE model: {str(e)}")
            return None
    
    def _load_feature_mappings(self):
        """Load feature mappings from CSV"""
        csv_path = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/FinanceLabeling/all_layers_financial_results/results_summary_all_layers_large_model_yahoo_finance.csv"
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            layer16_features = df[df['layer'] == 16].nlargest(20, 'f1_score')
            
            feature_mappings = {}
            for _, row in layer16_features.iterrows():
                feature_mappings[row['feature']] = row['label']
            
            return feature_mappings
        return {}
    
    def get_top_activations(self, text: str, top_n: int = 20) -> pd.DataFrame:
        """Get top SAE activations for a given text"""
        if self.sae_model is None:
            return pd.DataFrame()
        
        try:
            inputs = self.sae_model['tokenizer'](text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.sae_model['model'](**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[self.layer]  # Shape: [batch_size, seq_len, hidden_dim]
                
                # Reshape to [batch_size * seq_len, hidden_dim] for SAE encoding
                batch_size, seq_len, hidden_dim = hidden_states.shape
                hidden_states_flat = hidden_states.view(-1, hidden_dim)
                
                # Convert to same dtype as SAE weights
                hidden_states_flat = hidden_states_flat.to(self.sae_model['w_enc'].dtype)
                
                h = torch.matmul(hidden_states_flat, self.sae_model['w_enc'].T) + self.sae_model['b_enc']
                h_activated = torch.relu(h)
                
                # Reshape back to [batch_size, seq_len, sae_dim]
                h_activated = h_activated.view(batch_size, seq_len, -1)
                
                activations = h_activated.abs().sum(dim=1)  # Sum across sequence length
                top_activations = activations.topk(top_n)
                
                df = pd.DataFrame({
                    'feature_id': top_activations.indices[0].cpu().numpy(),
                    'activation': top_activations.values[0].cpu().numpy()
                })
                
                df['feature_label'] = df['feature_id'].apply(lambda x: self.feature_mappings.get(x, f"feature_{x}"))
                
                return df
        except Exception as e:
            logger.error(f"Error getting activations: {str(e)}")
            return pd.DataFrame()
    
    def analyze_headlines_for_features(self, headlines: List[str], top_n: int = 20) -> pd.DataFrame:
        """Analyze headlines and aggregate top activated features"""
        feature_sum_dict = {}
        feature_label_dict = {}
        
        for headline in headlines:
            activations_df = self.get_top_activations(headline, top_n=top_n)
            
            if activations_df.empty:
                continue
            
            for _, row in activations_df.iterrows():
                feature_id = row['feature_id']
                activation = row['activation']
                label = row['feature_label']
                
                if feature_id not in feature_sum_dict:
                    feature_sum_dict[feature_id] = activation
                    feature_label_dict[feature_id] = label
                else:
                    feature_sum_dict[feature_id] += activation
        
        aggregated_df = pd.DataFrame({
            'feature_id': list(feature_sum_dict.keys()),
            'total_activation': list(feature_sum_dict.values()),
            'feature_label': [feature_label_dict[f] for f in feature_sum_dict.keys()]
        })
        
        return aggregated_df.sort_values('total_activation', ascending=False)
    
    def build_feature_matrix(self, headlines: List[str], returns: List[float], selected_features: List[int]) -> pd.DataFrame:
        """Build feature matrix for trading analysis"""
        records = []
        
        for headline, ret in zip(headlines, returns):
            activations_df = self.get_top_activations(headline, top_n=50)
            
            if activations_df.empty:
                continue
            
            row_dict = {}
            
            for feature_id in selected_features:
                feature_name = f"feature_{feature_id}"
                if feature_id in activations_df['feature_id'].values:
                    row_dict[feature_name] = activations_df[activations_df['feature_id'] == feature_id]['activation'].iloc[0]
                else:
                    row_dict[feature_name] = 0.0
            
            row_dict['label'] = 1 if ret > 0.1 else 0
            row_dict['return'] = ret
            row_dict['headline'] = headline
            
            records.append(row_dict)
        
        return pd.DataFrame(records)

def main():
    """Main function"""
    logger.info("Starting Llama-2-7b-hf SAE Trading Analysis")
    
    analyzer = LlamaSAEMinimal()
    
    if analyzer.sae_model is None:
        logger.error("Failed to load SAE model. Exiting.")
        return
    
    # Use example financial headlines (GitHub data not accessible)
    logger.info("Using example financial headlines...")
    headlines = [
        "Apple reports record quarterly earnings with strong iPhone sales",
        "Federal Reserve raises interest rates by 0.25%",
        "Tesla stock surges 15% after positive earnings report",
        "Market volatility increases due to geopolitical tensions",
        "Bitcoin reaches new all-time high above $100,000",
        "Tech stocks face selling pressure amid inflation concerns",
        "Goldman Sachs upgrades Tesla to buy rating",
        "Oil prices fall 5% on demand concerns",
        "Amazon announces major cloud computing expansion",
        "Bank of America reports better-than-expected Q4 results"
    ]
    returns = [0.05, -0.02, 0.15, -0.08, 0.12, -0.03, 0.07, -0.05, 0.09, 0.04]
    
    # Analyze headlines for features
    logger.info("Analyzing headlines for features...")
    top_features_df = analyzer.analyze_headlines_for_features(headlines, top_n=20)
    
    logger.info(f"Found {len(top_features_df)} unique features")
    logger.info("Top 10 features:")
    for _, row in top_features_df.head(10).iterrows():
        logger.info(f"  {row['feature_label']}: {row['total_activation']:.4f}")
    
    # Build feature matrix
    selected_features = top_features_df.head(10)['feature_id'].tolist()
    feature_df = analyzer.build_feature_matrix(headlines, returns, selected_features)
    
    logger.info(f"Created feature matrix with {len(feature_df)} samples")
    
    # Save results to files
    logger.info("Saving results to files...")
    
    # Save top features
    top_features_df.to_csv('llama2_sae_top_features.csv', index=False)
    logger.info("Saved top features to: llama2_sae_top_features.csv")
    
    # Save feature matrix
    feature_df.to_csv('llama2_sae_feature_matrix.csv', index=False)
    logger.info("Saved feature matrix to: llama2_sae_feature_matrix.csv")
    
    # Save summary results
    summary = {
        'total_headlines': len(headlines),
        'total_features_found': len(top_features_df),
        'feature_matrix_samples': len(feature_df),
        'top_5_features': top_features_df.head(5).to_dict('records'),
        'feature_matrix_columns': list(feature_df.columns)
    }
    
    import json
    with open('llama2_sae_results_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved results summary to: llama2_sae_results_summary.json")
    
    logger.info("Analysis completed! Check the following files:")
    logger.info("- llama2_sae_top_features.csv")
    logger.info("- llama2_sae_feature_matrix.csv") 
    logger.info("- llama2_sae_results_summary.json")

if __name__ == "__main__":
    main()
