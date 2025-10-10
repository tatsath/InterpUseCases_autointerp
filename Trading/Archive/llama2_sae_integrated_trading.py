"""
Integrated Llama-2-7b-hf SAE Trading Analysis
Integrates with existing trading infrastructure and uses the notebook's approach
"""

import pandas as pd
import numpy as np
import torch
import os
import logging
from typing import Dict, List, Optional
from safetensors import safe_open
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaSAEIntegratedTrading:
    """Integrated trading analyzer using Llama-2-7b-hf SAE features"""
    
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
        
        # Initialize feature mappings for financial concepts
        self.feature_mappings = {
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
    
    def analyze_headlines_for_features(self, headlines: List[str], top_n: int = 10) -> pd.DataFrame:
        """
        Analyze headlines and aggregate top activated features
        
        Args:
            headlines: List of financial headlines
            top_n: Number of top features to return
            
        Returns:
            DataFrame with aggregated features
        """
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
        """
        Build feature matrix for trading analysis
        
        Args:
            headlines: List of financial headlines
            returns: List of corresponding returns
            selected_features: List of selected feature IDs
            
        Returns:
            DataFrame with feature matrix
        """
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
    
    def train_trading_model(self, df: pd.DataFrame) -> Dict:
        """
        Train a trading model using SAE features
        
        Args:
            df: DataFrame with SAE features and labels
            
        Returns:
            Dictionary with model results
        """
        # Prepare features
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
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
    """Main function to run the integrated trading analysis"""
    logger.info("Starting Integrated Llama-2-7b-hf SAE Trading Analysis")
    
    # Initialize analyzer
    analyzer = LlamaSAEIntegratedTrading()
    
    if analyzer.sae_model is None:
        logger.error("Failed to load SAE model. Exiting.")
        return
    
    # Example financial headlines (similar to the notebook)
    financial_headlines = [
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
    
    # Example returns (positive/negative)
    returns = [0.05, -0.02, 0.15, -0.08, 0.12, -0.03, 0.07, -0.05, 0.09, 0.04]
    
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
    
    # Step 4: Train trading model
    if len(feature_df) > 0:
        logger.info("Step 3: Training trading model...")
        model_results = analyzer.train_trading_model(feature_df)
        
        logger.info(f"Model accuracy: {model_results['accuracy']:.4f}")
        logger.info("Top 10 most important features:")
        for _, row in model_results['feature_importance'].head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Show classification report
        logger.info("Classification Report:")
        logger.info(f"Precision: {model_results['classification_report']['1']['precision']:.4f}")
        logger.info(f"Recall: {model_results['classification_report']['1']['recall']:.4f}")
        logger.info(f"F1-Score: {model_results['classification_report']['1']['f1-score']:.4f}")
    
    # Step 5: Analyze market sentiment
    logger.info("Step 4: Analyzing market sentiment...")
    sentiment_analysis = analyzer.analyze_market_sentiment(financial_headlines)
    
    if 'error' not in sentiment_analysis:
        logger.info(f"Analyzed {sentiment_analysis['total_texts']} texts")
        logger.info(f"Total activations: {sentiment_analysis['total_activations']}")
        logger.info(f"Total activation: {sentiment_analysis['total_activation']:.4f}")
        
        # Show sentiment by category
        logger.info("Sentiment by category:")
        for category, data in sentiment_analysis['sentiment_by_category'].items():
            if data['feature_count'] > 0:
                logger.info(f"  {category}: {data['total_activation']:.4f} total, {data['avg_activation']:.4f} avg")
    
    logger.info("Integrated trading analysis completed successfully!")

if __name__ == "__main__":
    main()
