"""
SAE Feature Processor for Finance Trading
Processes SAE features from the finance analysis and creates trading features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SAEFeatureProcessor:
    """Processes SAE features for finance trading applications"""
    
    def __init__(self):
        """Initialize the SAE feature processor with finance-specific features"""
        
        # Finance-related features from the README analysis
        # Organized by layer and feature importance
        self.finance_features = {
            'layer_4': {
                'earnings_reports': 127,      # F1: 0.72, Specialization: 7.61
                'valuation_changes': 141,     # F1: 0.28, Specialization: 7.58
                'performance_indicators': 1,  # F1: 0.365, Specialization: 7.03
                'stock_index_performance': 90, # F1: 0.84, Specialization: 5.63
                'inflation_indicators': 3,    # F1: 0.84, Specialization: 5.00
                'asset_diversification': 384, # F1: 0.64, Specialization: 4.53
                'private_equity': 2,          # F1: 0.808, Specialization: 4.05
                'foreign_exchange': 156,      # F1: 0.769, Specialization: 3.22
                'trading_strategies': 25,     # F1: 0.8, Specialization: 3.07
                'sector_innovations': 373     # F1: 0.08, Specialization: 2.90
            },
            'layer_10': {
                'earnings_reports': 384,      # F1: 0.288, Specialization: 14.50
                'cryptocurrency': 292,        # F1: 0.673, Specialization: 5.14
                'revenue_metrics': 273,       # F1: 0.9, Specialization: 5.02
                'stock_performance': 173,     # F1: 0.865, Specialization: 4.35
                'economic_indicators': 343,   # F1: 0.154, Specialization: 4.35
                'asset_diversification': 372, # F1: 0.84, Specialization: 3.39
                'private_equity': 17,         # F1: 0.7, Specialization: 3.34
                'foreign_exchange': 389,      # F1: 0.82, Specialization: 3.11
                'trading_strategies': 303,    # F1: 0.808, Specialization: 2.63
                'fintech_innovation': 47      # F1: 0.615, Specialization: 2.54
            },
            'layer_16': {
                'earnings_reports': 332,      # F1: 0.769, Specialization: 19.56
                'major_figures': 105,         # F1: 0.692, Specialization: 9.58
                'revenue_metrics': 214,       # F1: 0.76, Specialization: 8.85
                'stock_index_metrics': 66,    # F1: 0.577, Specialization: 4.75
                'inflation_labor': 181,       # F1: 0.635, Specialization: 4.65
                'portfolio_diversification': 203, # F1: 0.22, Specialization: 4.33
                'private_equity': 340,        # F1: 0.538, Specialization: 4.09
                'central_bank_policies': 162, # F1: 0.76, Specialization: 3.27
                'trading_strategies': 267,    # F1: 0.269, Specialization: 3.26
                'sector_investment': 133      # F1: 0.8, Specialization: 3.19
            },
            'layer_22': {
                'earnings_reports': 396,      # F1: 0.462, Specialization: 16.70
                'value_milestones': 353,      # F1: 0.42, Specialization: 11.06
                'performance_metrics': 220,   # F1: 0.76, Specialization: 6.69
                'performance_updates': 184,   # F1: 0.808, Specialization: 5.04
                'inflation_indicators': 276,  # F1: 0.68, Specialization: 4.81
                'asset_diversification': 83,  # F1: 0.731, Specialization: 3.75
                'private_equity': 303,        # F1: 0.34, Specialization: 3.54
                'central_bank_policies': 387, # F1: 0.654, Specialization: 3.44
                'trading_strategies': 239,    # F1: 0.712, Specialization: 3.38
                'fintech_solutions': 101      # F1: 0.76, Specialization: 3.20
            },
            'layer_28': {
                'earnings_reports': 262,      # F1: 0.5, Specialization: 21.74
                'value_changes': 27,          # F1: 0.36, Specialization: 12.73
                'revenue_figures': 181,       # F1: 0.808, Specialization: 6.03
                'stock_index_performance': 171, # F1: 0.06, Specialization: 4.88
                'inflation_indicators': 154,  # F1: 0.269, Specialization: 4.46
                'portfolio_diversification': 83, # F1: 0.865, Specialization: 4.24
                'private_equity': 389,        # F1: 0.615, Specialization: 4.13
                'currency_volatility': 172,   # F1: 0.096, Specialization: 3.79
                'trading_strategies': 333,    # F1: 0.52, Specialization: 3.57
                'sector_investment': 350      # F1: 0.788, Specialization: 3.53
            }
        }
        
        # Feature weights based on F1 scores and specialization
        self.feature_weights = self._calculate_feature_weights()
        
    def _calculate_feature_weights(self) -> Dict[str, float]:
        """Calculate feature weights based on F1 scores and specialization"""
        weights = {}
        
        for layer, features in self.finance_features.items():
            for feature_name, feature_id in features.items():
                # Get F1 score and specialization from the analysis
                f1_score = self._get_f1_score(layer, feature_name)
                specialization = self._get_specialization(layer, feature_name)
                
                # Weight = F1 * log(specialization + 1) to balance performance and specificity
                weight = f1_score * np.log(specialization + 1)
                weights[f"{layer}_{feature_name}"] = weight
        
        return weights
    
    def _get_f1_score(self, layer: str, feature_name: str) -> float:
        """Get F1 score for a feature (from README analysis)"""
        f1_scores = {
            'layer_4': {
                'earnings_reports': 0.72, 'valuation_changes': 0.28, 'performance_indicators': 0.365,
                'stock_index_performance': 0.84, 'inflation_indicators': 0.84, 'asset_diversification': 0.64,
                'private_equity': 0.808, 'foreign_exchange': 0.769, 'trading_strategies': 0.8, 'sector_innovations': 0.08
            },
            'layer_10': {
                'earnings_reports': 0.288, 'cryptocurrency': 0.673, 'revenue_metrics': 0.9, 'stock_performance': 0.865,
                'economic_indicators': 0.154, 'asset_diversification': 0.84, 'private_equity': 0.7,
                'foreign_exchange': 0.82, 'trading_strategies': 0.808, 'fintech_innovation': 0.615
            },
            'layer_16': {
                'earnings_reports': 0.769, 'major_figures': 0.692, 'revenue_metrics': 0.76, 'stock_index_metrics': 0.577,
                'inflation_labor': 0.635, 'portfolio_diversification': 0.22, 'private_equity': 0.538,
                'central_bank_policies': 0.76, 'trading_strategies': 0.269, 'sector_investment': 0.8
            },
            'layer_22': {
                'earnings_reports': 0.462, 'value_milestones': 0.42, 'performance_metrics': 0.76, 'performance_updates': 0.808,
                'inflation_indicators': 0.68, 'asset_diversification': 0.731, 'private_equity': 0.34,
                'central_bank_policies': 0.654, 'trading_strategies': 0.712, 'fintech_solutions': 0.76
            },
            'layer_28': {
                'earnings_reports': 0.5, 'value_changes': 0.36, 'revenue_figures': 0.808, 'stock_index_performance': 0.06,
                'inflation_indicators': 0.269, 'portfolio_diversification': 0.865, 'private_equity': 0.615,
                'currency_volatility': 0.096, 'trading_strategies': 0.52, 'sector_investment': 0.788
            }
        }
        
        return f1_scores.get(layer, {}).get(feature_name, 0.5)
    
    def _get_specialization(self, layer: str, feature_name: str) -> float:
        """Get specialization score for a feature (from README analysis)"""
        specializations = {
            'layer_4': {
                'earnings_reports': 7.61, 'valuation_changes': 7.58, 'performance_indicators': 7.03,
                'stock_index_performance': 5.63, 'inflation_indicators': 5.00, 'asset_diversification': 4.53,
                'private_equity': 4.05, 'foreign_exchange': 3.22, 'trading_strategies': 3.07, 'sector_innovations': 2.90
            },
            'layer_10': {
                'earnings_reports': 14.50, 'cryptocurrency': 5.14, 'revenue_metrics': 5.02, 'stock_performance': 4.35,
                'economic_indicators': 4.35, 'asset_diversification': 3.39, 'private_equity': 3.34,
                'foreign_exchange': 3.11, 'trading_strategies': 2.63, 'fintech_innovation': 2.54
            },
            'layer_16': {
                'earnings_reports': 19.56, 'major_figures': 9.58, 'revenue_metrics': 8.85, 'stock_index_metrics': 4.75,
                'inflation_labor': 4.65, 'portfolio_diversification': 4.33, 'private_equity': 4.09,
                'central_bank_policies': 3.27, 'trading_strategies': 3.26, 'sector_investment': 3.19
            },
            'layer_22': {
                'earnings_reports': 16.70, 'value_milestones': 11.06, 'performance_metrics': 6.69, 'performance_updates': 5.04,
                'inflation_indicators': 4.81, 'asset_diversification': 3.75, 'private_equity': 3.54,
                'central_bank_policies': 3.44, 'trading_strategies': 3.38, 'fintech_solutions': 3.20
            },
            'layer_28': {
                'earnings_reports': 21.74, 'value_changes': 12.73, 'revenue_figures': 6.03, 'stock_index_performance': 4.88,
                'inflation_indicators': 4.46, 'portfolio_diversification': 4.24, 'private_equity': 4.13,
                'currency_volatility': 3.79, 'trading_strategies': 3.57, 'sector_investment': 3.53
            }
        }
        
        return specializations.get(layer, {}).get(feature_name, 3.0)
    
    def create_sae_features(self, 
                          market_data: pd.DataFrame,
                          sae_activations: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create SAE-based trading features from market data
        
        Args:
            market_data: DataFrame with OHLCV data
            sae_activations: Optional DataFrame with actual SAE activations
            
        Returns:
            DataFrame with SAE-based features
        """
        df = market_data.copy()
        
        if sae_activations is not None:
            # Use actual SAE activations if provided
            return self._process_actual_sae_activations(df, sae_activations)
        else:
            # Create synthetic SAE features based on market patterns
            return self._create_synthetic_sae_features(df)
    
    def _process_actual_sae_activations(self, 
                                      market_data: pd.DataFrame, 
                                      sae_activations: pd.DataFrame) -> pd.DataFrame:
        """Process actual SAE activations for trading features"""
        df = market_data.copy()
        
        # Merge SAE activations with market data
        sae_features = []
        
        for layer, features in self.finance_features.items():
            for feature_name, feature_id in features.items():
                col_name = f"SAE_{layer}_{feature_name}"
                
                if col_name in sae_activations.columns:
                    df[col_name] = sae_activations[col_name]
                    sae_features.append(col_name)
                else:
                    # Create synthetic feature if not available
                    df[col_name] = self._create_synthetic_feature(df, layer, feature_name)
                    sae_features.append(col_name)
        
        # Add composite features
        df = self._add_composite_features(df, sae_features)
        
        return df
    
    def _create_synthetic_sae_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Create synthetic SAE features based on market patterns"""
        df = market_data.copy()
        sae_features = []
        
        for layer, features in self.finance_features.items():
            for feature_name, feature_id in features.items():
                col_name = f"SAE_{layer}_{feature_name}"
                df[col_name] = self._create_synthetic_feature(df, layer, feature_name)
                sae_features.append(col_name)
        
        # Add composite features
        df = self._add_composite_features(df, sae_features)
        
        return df
    
    def _create_synthetic_feature(self, df: pd.DataFrame, layer: str, feature_name: str) -> pd.Series:
        """Create a synthetic SAE feature based on market patterns"""
        price = df['close']
        returns = df['returns']
        volume = df['volume']
        
        # Base activation level
        base_activation = np.random.normal(0, 0.1, len(df))
        
        # Add market-based patterns
        if 'earnings' in feature_name or 'revenue' in feature_name:
            # Correlate with price momentum
            momentum = price.pct_change(20).fillna(0)
            activation = base_activation + 0.3 * np.tanh(momentum)
            
        elif 'volatility' in feature_name or 'volatility' in feature_name:
            # Correlate with volatility
            vol = returns.rolling(20).std().fillna(0)
            activation = base_activation + 0.4 * np.tanh(vol * 100)
            
        elif 'trading' in feature_name or 'strategies' in feature_name:
            # Correlate with volume and returns
            vol_norm = volume.rolling(20).mean().fillna(0)
            activation = base_activation + 0.2 * np.tanh(returns * 10) + 0.1 * np.tanh(vol_norm / vol_norm.mean())
            
        elif 'inflation' in feature_name or 'economic' in feature_name:
            # Correlate with longer-term trends
            trend = price.rolling(50).mean().pct_change().fillna(0)
            activation = base_activation + 0.3 * np.tanh(trend * 5)
            
        elif 'private_equity' in feature_name or 'venture' in feature_name:
            # Correlate with high-volume periods
            vol_spike = (volume / volume.rolling(20).mean()).fillna(1)
            activation = base_activation + 0.2 * np.tanh(vol_spike - 1)
            
        else:
            # Generic pattern based on returns and volume
            activation = base_activation + 0.1 * np.tanh(returns * 5) + 0.1 * np.tanh(volume / volume.rolling(20).mean() - 1)
        
        # Apply layer-specific scaling
        layer_scaling = {'layer_4': 1.0, 'layer_10': 1.2, 'layer_16': 1.5, 'layer_22': 1.8, 'layer_28': 2.0}
        activation = activation * layer_scaling.get(layer, 1.0)
        
        # Ensure non-negative activations (SAE features are typically non-negative)
        activation = np.maximum(activation, 0)
        
        return pd.Series(activation, index=df.index)
    
    def _add_composite_features(self, df: pd.DataFrame, sae_features: List[str]) -> pd.DataFrame:
        """Add composite SAE features"""
        
        # Layer-wise aggregations
        for layer in ['layer_4', 'layer_10', 'layer_16', 'layer_22', 'layer_28']:
            layer_features = [f for f in sae_features if f.startswith(f'SAE_{layer}_')]
            if layer_features:
                df[f'SAE_{layer}_composite'] = df[layer_features].mean(axis=1)
                df[f'SAE_{layer}_max'] = df[layer_features].max(axis=1)
                df[f'SAE_{layer}_std'] = df[layer_features].std(axis=1)
        
        # Cross-layer features
        all_layer_composites = [f'SAE_{layer}_composite' for layer in ['layer_4', 'layer_10', 'layer_16', 'layer_22', 'layer_28']]
        if all_layer_composites:
            df['SAE_all_layers_mean'] = df[all_layer_composites].mean(axis=1)
            df['SAE_all_layers_std'] = df[all_layer_composites].std(axis=1)
            df['SAE_all_layers_max'] = df[all_layer_composites].max(axis=1)
        
        # Feature type aggregations
        feature_types = {
            'earnings': [f for f in sae_features if 'earnings' in f or 'revenue' in f],
            'volatility': [f for f in sae_features if 'volatility' in f or 'volatility' in f],
            'trading': [f for f in sae_features if 'trading' in f or 'strategies' in f],
            'economic': [f for f in sae_features if 'inflation' in f or 'economic' in f or 'central_bank' in f]
        }
        
        for ftype, features in feature_types.items():
            if features:
                df[f'SAE_{ftype}_composite'] = df[features].mean(axis=1)
        
        return df
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores"""
        importance_data = []
        
        for layer, features in self.finance_features.items():
            for feature_name, feature_id in features.items():
                f1_score = self._get_f1_score(layer, feature_name)
                specialization = self._get_specialization(layer, feature_name)
                weight = self.feature_weights[f"{layer}_{feature_name}"]
                
                importance_data.append({
                    'layer': layer,
                    'feature_name': feature_name,
                    'feature_id': feature_id,
                    'f1_score': f1_score,
                    'specialization': specialization,
                    'weight': weight
                })
        
        return pd.DataFrame(importance_data).sort_values('weight', ascending=False)
    
    def save_sae_features(self, df: pd.DataFrame, filename: str, data_dir: str = 'data'):
        """Save SAE features to parquet file"""
        os.makedirs(data_dir, exist_ok=True)
        filepath = os.path.join(data_dir, filename)
        df.to_parquet(filepath)
        logger.info(f"SAE features saved to {filepath}")

def main():
    """Example usage of the SAE feature processor"""
    processor = SAEFeatureProcessor()
    
    # Get feature importance
    importance_df = processor.get_feature_importance()
    print("Feature Importance (Top 10):")
    print(importance_df.head(10))
    
    # Create sample market data
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    sample_data = pd.DataFrame({
        'close': 50000 + np.cumsum(np.random.randn(1000) * 100),
        'volume': np.random.exponential(1000, 1000),
        'returns': np.random.randn(1000) * 0.01
    }, index=dates)
    
    # Create SAE features
    sae_df = processor.create_sae_features(sample_data)
    print(f"\nCreated SAE features: {sae_df.shape}")
    print(f"SAE feature columns: {[col for col in sae_df.columns if col.startswith('SAE_')][:10]}")

if __name__ == "__main__":
    main()
