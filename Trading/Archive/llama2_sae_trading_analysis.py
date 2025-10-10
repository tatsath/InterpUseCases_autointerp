"""
Llama-2-7b-hf SAE Trading Analysis
Combines SAE features from Llama-2-7b-hf model with trading analysis
Based on the ActivationDetectTwoPass.ipynb notebook example
"""

import pandas as pd
import numpy as np
import torch
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import requests
from safetensors import safe_open

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaSAETradingAnalyzer:
    """Trading analyzer using Llama-2-7b-hf SAE features"""
    
    def __init__(self, 
                 model_path: str = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun",
                 layer: int = 16,
                 device: str = "cuda:0"):
        """
        Initialize the analyzer
        
        Args:
            model_path: Path to the trained SAE model
            layer: Layer number to use (16)
            device: Device to use for computation
        """
        self.model_path = model_path
        self.layer = layer
        self.device = device
        
        # Load the SAE model
        self.sae_model = self._load_sae_model()
        
        # Load the base Llama model
        self.base_model = self._load_base_model()
        
        # Initialize feature mappings
        self.feature_mappings = self._initialize_feature_mappings()
        
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
    
    def _load_base_model(self):
        """Load the base Llama model"""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            model_name = "meta-llama/Llama-2-7b-hf"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16, device_map=self.device)
            
            return {
                'model': model,
                'tokenizer': tokenizer
            }
        except Exception as e:
            logger.error(f"Error loading base model: {str(e)}")
            return None
    
    def _initialize_feature_mappings(self):
        """Initialize feature mappings for financial concepts"""
        return {
            'earnings_reports': [332, 105, 214],  # Top features for earnings
            'revenue_metrics': [214, 66, 181],    # Revenue-related features
            'stock_performance': [66, 133, 267],  # Stock performance features
            'trading_strategies': [267, 133, 340], # Trading strategy features
            'economic_indicators': [181, 162, 203], # Economic indicators
            'volatility': [162, 203, 133],        # Volatility features
            'market_sentiment': [340, 267, 162],   # Market sentiment features
            'financial_news': [332, 214, 105],     # Financial news features
            'risk_assessment': [203, 162, 340],   # Risk assessment features
            'portfolio_management': [133, 203, 267] # Portfolio management features
        }
    
    def get_top_activations(self, text: str, top_n: int = 20) -> pd.DataFrame:
        """
        Get top SAE activations for a given text
        
        Args:
            text: Input text to analyze
            top_n: Number of top activations to return
            
        Returns:
            DataFrame with top activations
        """
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
                hidden_states = outputs.hidden_states[self.layer]  # Get layer 16 activations
                
                # Encode through SAE
                h = torch.matmul(hidden_states, self.sae_model['w_enc']) + self.sae_model['b_enc']
                h_activated = torch.relu(h)  # ReLU activation
                
                # Get top activations
                activations = h_activated.abs().sum(dim=1)  # Sum across sequence length
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
        # Map feature IDs to financial concepts
        for concept, features in self.feature_mappings.items():
            if feature_id in features:
                return f"{concept}_{feature_id}"
        
        return f"feature_{feature_id}"
    
    def analyze_financial_text(self, text: str) -> Dict:
        """
        Analyze financial text using SAE features
        
        Args:
            text: Financial text to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Get top activations
        activations_df = self.get_top_activations(text, top_n=50)
        
        if activations_df.empty:
            return {"error": "No activations found"}
        
        # Analyze feature categories
        category_analysis = {}
        for category, feature_ids in self.feature_mappings.items():
            category_features = activations_df[activations_df['feature_id'].isin(feature_ids)]
            if not category_features.empty:
                category_analysis[category] = {
                    'total_activation': category_features['activation'].sum(),
                    'max_activation': category_features['activation'].max(),
                    'feature_count': len(category_features),
                    'top_features': category_features.nlargest(3, 'activation')[['feature_id', 'activation']].to_dict('records')
                }
        
        # Get overall top features
        top_features = activations_df.nlargest(10, 'activation')
        
        return {
            'text': text,
            'total_activations': len(activations_df),
            'max_activation': activations_df['activation'].max(),
            'category_analysis': category_analysis,
            'top_features': top_features.to_dict('records')
        }
    
    def create_trading_dataset(self, headlines: List[str], returns: List[float]) -> pd.DataFrame:
        """
        Create a trading dataset using SAE features
        
        Args:
            headlines: List of financial headlines
            returns: List of corresponding returns
            
        Returns:
            DataFrame with SAE features and labels
        """
        if len(headlines) != len(returns):
            raise ValueError("Headlines and returns must have the same length")
        
        # Create feature matrix
        feature_data = []
        
        for i, (headline, ret) in enumerate(zip(headlines, returns)):
            logger.info(f"Processing headline {i+1}/{len(headlines)}")
            
            # Get activations for this headline
            activations_df = self.get_top_activations(headline, top_n=100)
            
            if activations_df.empty:
                logger.warning(f"No activations found for headline: {headline}")
                continue
            
            # Create feature vector
            feature_vector = {}
            
            # Add individual feature activations
            for _, row in activations_df.iterrows():
                feature_vector[f"feature_{row['feature_id']}"] = row['activation']
            
            # Add category-based features
            for category, feature_ids in self.feature_mappings.items():
                category_features = activations_df[activations_df['feature_id'].isin(feature_ids)]
                if not category_features.empty:
                    feature_vector[f"{category}_total"] = category_features['activation'].sum()
                    feature_vector[f"{category}_max"] = category_features['activation'].max()
                    feature_vector[f"{category}_count"] = len(category_features)
                else:
                    feature_vector[f"{category}_total"] = 0.0
                    feature_vector[f"{category}_max"] = 0.0
                    feature_vector[f"{category}_count"] = 0
            
            # Add label
            feature_vector['label'] = 1 if ret > 0.1 else 0  # Binary classification
            feature_vector['return'] = ret
            feature_vector['headline'] = headline
            
            feature_data.append(feature_vector)
        
        return pd.DataFrame(feature_data)
    
    def train_trading_model(self, df: pd.DataFrame) -> Dict:
        """
        Train a trading model using SAE features
        
        Args:
            df: DataFrame with SAE features and labels
            
        Returns:
            Dictionary with model results
        """
        # Prepare features
        feature_cols = [col for col in df.columns if col.startswith('feature_') or col.endswith('_total') or col.endswith('_max') or col.endswith('_count')]
        X = df[feature_cols].fillna(0)
        y = df['label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            'model': model,
            'accuracy': accuracy,
            'feature_importance': feature_importance,
            'classification_report': report,
            'test_predictions': y_pred,
            'test_probabilities': y_pred_proba,
            'test_labels': y_test
        }
    
    def analyze_market_sentiment(self, texts: List[str]) -> Dict:
        """
        Analyze market sentiment across multiple texts
        
        Args:
            texts: List of financial texts
            
        Returns:
            Dictionary with sentiment analysis
        """
        all_activations = []
        
        for text in texts:
            activations_df = self.get_top_activations(text, top_n=50)
            if not activations_df.empty:
                all_activations.append(activations_df)
        
        if not all_activations:
            return {"error": "No activations found"}
        
        # Combine all activations
        combined_df = pd.concat(all_activations, ignore_index=True)
        
        # Analyze by category
        sentiment_analysis = {}
        for category, feature_ids in self.feature_mappings.items():
            category_features = combined_df[combined_df['feature_id'].isin(feature_ids)]
            if not category_features.empty:
                sentiment_analysis[category] = {
                    'total_activation': category_features['activation'].sum(),
                    'avg_activation': category_features['activation'].mean(),
                    'max_activation': category_features['activation'].max(),
                    'feature_count': len(category_features)
                }
        
        # Get overall sentiment score
        total_activation = combined_df['activation'].sum()
        max_activation = combined_df['activation'].max()
        
        return {
            'total_texts': len(texts),
            'total_activations': len(combined_df),
            'total_activation': total_activation,
            'max_activation': max_activation,
            'sentiment_by_category': sentiment_analysis,
            'top_features': combined_df.nlargest(20, 'activation').to_dict('records')
        }

def main():
    """Main function to run the trading analysis"""
    logger.info("Starting Llama-2-7b-hf SAE Trading Analysis")
    
    # Initialize analyzer
    analyzer = LlamaSAETradingAnalyzer()
    
    if analyzer.sae_model is None:
        logger.error("Failed to load SAE model. Exiting.")
        return
    
    # Example financial texts
    financial_texts = [
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
    
    # Analyze individual texts
    logger.info("Analyzing individual financial texts...")
    for i, text in enumerate(financial_texts[:3]):  # Analyze first 3 texts
        logger.info(f"Analyzing text {i+1}: {text[:50]}...")
        analysis = analyzer.analyze_financial_text(text)
        
        if 'error' not in analysis:
            logger.info(f"Found {analysis['total_activations']} activations")
            logger.info(f"Max activation: {analysis['max_activation']:.4f}")
            
            # Show top categories
            for category, data in analysis['category_analysis'].items():
                if data['feature_count'] > 0:
                    logger.info(f"  {category}: {data['total_activation']:.4f} total activation")
    
    # Create trading dataset
    logger.info("Creating trading dataset...")
    returns = [0.05, -0.02, 0.15, -0.08, 0.12, -0.03, 0.07, -0.05, 0.09, 0.04]  # Example returns
    
    trading_df = analyzer.create_trading_dataset(financial_texts, returns)
    logger.info(f"Created trading dataset with {len(trading_df)} samples and {len(trading_df.columns)} features")
    
    # Train trading model
    if len(trading_df) > 0:
        logger.info("Training trading model...")
        model_results = analyzer.train_trading_model(trading_df)
        
        logger.info(f"Model accuracy: {model_results['accuracy']:.4f}")
        logger.info("Top 10 most important features:")
        for _, row in model_results['feature_importance'].head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Analyze market sentiment
    logger.info("Analyzing market sentiment...")
    sentiment_analysis = analyzer.analyze_market_sentiment(financial_texts)
    
    if 'error' not in sentiment_analysis:
        logger.info(f"Analyzed {sentiment_analysis['total_texts']} texts")
        logger.info(f"Total activations: {sentiment_analysis['total_activations']}")
        logger.info(f"Total activation: {sentiment_analysis['total_activation']:.4f}")
        
        # Show sentiment by category
        for category, data in sentiment_analysis['sentiment_by_category'].items():
            if data['feature_count'] > 0:
                logger.info(f"  {category}: {data['total_activation']:.4f} total, {data['avg_activation']:.4f} avg")
    
    logger.info("Analysis completed successfully!")

if __name__ == "__main__":
    main()
