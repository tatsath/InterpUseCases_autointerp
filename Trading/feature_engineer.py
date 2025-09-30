"""
Feature Engineering for Trading
Creates baseline technical features and integrates SAE features for trading
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Creates and processes features for trading models"""
    
    def __init__(self):
        """Initialize the feature engineer"""
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.feature_selector = None
        self.feature_names = []
        
    def create_baseline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create baseline technical analysis features
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional technical features
        """
        df = df.copy()
        
        # Price-based features
        df = self._add_price_features(df)
        
        # Volume features
        df = self._add_volume_features(df)
        
        # Momentum features
        df = self._add_momentum_features(df)
        
        # Volatility features
        df = self._add_volatility_features(df)
        
        # Trend features
        df = self._add_trend_features(df)
        
        # Market microstructure features
        df = self._add_microstructure_features(df)
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based technical features"""
        
        # Basic price ratios
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        df['high_close_ratio'] = df['high'] / df['close']
        df['low_close_ratio'] = df['low'] / df['close']
        
        # Price position within the bar
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        
        # Price gaps
        df['gap_up'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        df['gap_down'] = (df['close'].shift(1) - df['open']) / df['close'].shift(1)
        
        # Price ranges
        df['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                np.abs(df['high'] - df['close'].shift(1)),
                np.abs(df['low'] - df['close'].shift(1))
            )
        )
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        
        # Volume ratios
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_std_ratio'] = df['volume'] / df['volume'].rolling(20).std()
        
        # Volume-price relationship
        df['volume_price_trend'] = df['volume'] * df['returns']
        df['volume_weighted_price'] = (df['volume'] * df['close']).rolling(20).sum() / df['volume'].rolling(20).sum()
        
        # Volume momentum
        df['volume_momentum'] = df['volume'].pct_change(5)
        df['volume_acceleration'] = df['volume_momentum'].diff()
        
        # On-balance volume
        df['obv'] = (df['volume'] * np.sign(df['returns'])).cumsum()
        df['obv_ma'] = df['obv'].rolling(20).mean()
        df['obv_ratio'] = df['obv'] / df['obv_ma']
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features"""
        
        # Simple moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'price_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
        
        # Exponential moving averages
        for period in [5, 10, 20, 50]:
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            df[f'price_ema_{period}_ratio'] = df['close'] / df[f'ema_{period}']
        
        # Moving average crossovers
        df['sma_cross_5_20'] = (df['sma_5'] > df['sma_20']).astype(int)
        df['ema_cross_5_20'] = (df['ema_5'] > df['ema_20']).astype(int)
        df['ema_cross_10_50'] = (df['ema_10'] > df['ema_50']).astype(int)
        
        # RSI (Relative Strength Index)
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        df['rsi_21'] = self._calculate_rsi(df['close'], 21)
        
        # MACD
        macd_line, signal_line, histogram = self._calculate_macd(df['close'])
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        
        # Stochastic Oscillator
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df['high'], df['low'], df['close'])
        
        # Williams %R
        df['williams_r'] = self._calculate_williams_r(df['high'], df['low'], df['close'])
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features"""
        
        # Rolling volatility
        for period in [5, 10, 20, 50]:
            df[f'volatility_{period}'] = df['returns'].rolling(period).std()
            df[f'volatility_{period}_norm'] = df[f'volatility_{period}'] / df[f'volatility_{period}'].rolling(50).mean()
        
        # Average True Range (ATR)
        df['atr_14'] = df['true_range'].rolling(14).mean()
        df['atr_21'] = df['true_range'].rolling(21).mean()
        df['atr_ratio'] = df['atr_14'] / df['atr_21']
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['close'])
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
        
        # Volatility regime
        df['volatility_regime'] = (df['volatility_20'] > df['volatility_20'].rolling(50).quantile(0.8)).astype(int)
        
        return df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-based features"""
        
        # Linear regression slope
        for period in [10, 20, 50]:
            df[f'trend_slope_{period}'] = df['close'].rolling(period).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == period else np.nan
            )
        
        # ADX (Average Directional Index)
        df['adx'] = self._calculate_adx(df['high'], df['low'], df['close'])
        
        # Parabolic SAR
        df['sar'] = self._calculate_sar(df['high'], df['low'], df['close'])
        df['sar_signal'] = (df['close'] > df['sar']).astype(int)
        
        # Trend strength
        df['trend_strength'] = df['close'].rolling(20).apply(
            lambda x: len([i for i in range(1, len(x)) if x.iloc[i] > x.iloc[i-1]]) / len(x)
        )
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        
        # Bid-ask spread proxy (using high-low as proxy)
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        
        # Tick-by-tick features (using minute data as proxy)
        df['price_impact'] = df['returns'] / (df['volume'] + 1e-8)
        df['volume_impact'] = df['volume'] * np.abs(df['returns'])
        
        # Market efficiency proxy
        df['efficiency_ratio'] = np.abs(df['close'] - df['close'].shift(10)) / df['true_range'].rolling(10).sum()
        
        # Order flow imbalance proxy
        df['order_flow_proxy'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
        
        return df
    
    def integrate_sae_features(self, 
                             baseline_df: pd.DataFrame, 
                             sae_df: pd.DataFrame) -> pd.DataFrame:
        """
        Integrate SAE features with baseline features
        
        Args:
            baseline_df: DataFrame with baseline technical features
            sae_df: DataFrame with SAE features
            
        Returns:
            Combined DataFrame with all features
        """
        # Merge on timestamp and symbol
        if 'symbol' in baseline_df.columns and 'symbol' in sae_df.columns:
            combined_df = baseline_df.merge(
                sae_df, 
                left_index=True, 
                right_index=True, 
                how='left'
            )
        else:
            combined_df = baseline_df.merge(
                sae_df, 
                left_index=True, 
                right_index=True, 
                how='left'
            )
        
        # Fill missing SAE features with 0
        sae_columns = [col for col in combined_df.columns if col.startswith('SAE_')]
        combined_df[sae_columns] = combined_df[sae_columns].fillna(0)
        
        # Create interaction features between SAE and technical features
        combined_df = self._create_interaction_features(combined_df)
        
        return combined_df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between SAE and technical features"""
        
        # SAE features
        sae_columns = [col for col in df.columns if col.startswith('SAE_')]
        
        # Key technical features for interactions
        tech_features = ['rsi_14', 'macd', 'bb_position', 'volume_ma_ratio', 'volatility_20']
        available_tech = [f for f in tech_features if f in df.columns]
        
        # Create interactions between SAE and technical features
        for sae_col in sae_columns[:10]:  # Limit to top 10 SAE features
            for tech_col in available_tech:
                if sae_col in df.columns and tech_col in df.columns:
                    interaction_name = f"{sae_col}_x_{tech_col}"
                    df[interaction_name] = df[sae_col] * df[tech_col]
        
        return df
    
    def prepare_features_for_model(self, 
                                 df: pd.DataFrame, 
                                 target_col: str = 'returns',
                                 feature_selection: bool = True,
                                 n_features: int = 50) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for machine learning model
        
        Args:
            df: DataFrame with all features
            target_col: Target column name
            feature_selection: Whether to perform feature selection
            n_features: Number of features to select
            
        Returns:
            Tuple of (features_df, feature_names)
        """
        # Remove non-feature columns
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'symbol', 'returns', 'log_returns']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove columns with too many NaN values
        nan_threshold = 0.5
        valid_cols = []
        for col in feature_cols:
            if df[col].isna().sum() / len(df) < nan_threshold:
                valid_cols.append(col)
        
        # Create features DataFrame
        features_df = df[valid_cols].copy()
        
        # Fill remaining NaN values
        features_df = features_df.fillna(features_df.median())
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features_df)
        features_df = pd.DataFrame(features_scaled, columns=valid_cols, index=df.index)
        
        # Feature selection
        if feature_selection and target_col in df.columns:
            # Remove rows with NaN target
            valid_idx = df[target_col].notna()
            X = features_df[valid_idx]
            y = df.loc[valid_idx, target_col]
            
            # Select best features
            self.feature_selector = SelectKBest(f_regression, k=min(n_features, len(valid_cols)))
            X_selected = self.feature_selector.fit_transform(X, y)
            
            # Get selected feature names
            selected_mask = self.feature_selector.get_support()
            self.feature_names = [col for col, selected in zip(valid_cols, selected_mask) if selected]
            
            # Create final features DataFrame
            features_df = pd.DataFrame(X_selected, columns=self.feature_names, index=X.index)
        else:
            self.feature_names = valid_cols
        
        return features_df, self.feature_names
    
    def get_feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance from trained model"""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        elif hasattr(model, 'coef_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(model.coef_[0])
            }).sort_values('importance', ascending=False)
        else:
            logger.warning("Model does not have feature importance or coefficients")
            return pd.DataFrame()
        
        return importance_df
    
    # Technical indicator calculation methods
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3):
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def _calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
        """Calculate Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
        """Calculate ADX (simplified version)"""
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - close.shift(1)),
                np.abs(low - close.shift(1))
            )
        )
        atr = tr.rolling(window=period).mean()
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def _calculate_sar(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                      acceleration: float = 0.02, maximum: float = 0.2):
        """Calculate Parabolic SAR (simplified version)"""
        sar = pd.Series(index=close.index, dtype=float)
        trend = pd.Series(index=close.index, dtype=int)
        af = pd.Series(index=close.index, dtype=float)
        
        # Initialize
        sar.iloc[0] = low.iloc[0]
        trend.iloc[0] = 1
        af.iloc[0] = acceleration
        
        for i in range(1, len(close)):
            if trend.iloc[i-1] == 1:  # Uptrend
                sar.iloc[i] = sar.iloc[i-1] + af.iloc[i-1] * (high.iloc[i-1] - sar.iloc[i-1])
                if low.iloc[i] <= sar.iloc[i]:
                    trend.iloc[i] = -1
                    sar.iloc[i] = high.iloc[i-1]
                    af.iloc[i] = acceleration
                else:
                    trend.iloc[i] = 1
                    if high.iloc[i] > high.iloc[i-1]:
                        af.iloc[i] = min(af.iloc[i-1] + acceleration, maximum)
                    else:
                        af.iloc[i] = af.iloc[i-1]
            else:  # Downtrend
                sar.iloc[i] = sar.iloc[i-1] + af.iloc[i-1] * (low.iloc[i-1] - sar.iloc[i-1])
                if high.iloc[i] >= sar.iloc[i]:
                    trend.iloc[i] = 1
                    sar.iloc[i] = low.iloc[i-1]
                    af.iloc[i] = acceleration
                else:
                    trend.iloc[i] = -1
                    if low.iloc[i] < low.iloc[i-1]:
                        af.iloc[i] = min(af.iloc[i-1] + acceleration, maximum)
                    else:
                        af.iloc[i] = af.iloc[i-1]
        
        return sar

def main():
    """Example usage of the feature engineer"""
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    sample_data = pd.DataFrame({
        'open': 50000 + np.cumsum(np.random.randn(1000) * 10),
        'high': 50000 + np.cumsum(np.random.randn(1000) * 10) + np.random.exponential(50, 1000),
        'low': 50000 + np.cumsum(np.random.randn(1000) * 10) - np.random.exponential(50, 1000),
        'close': 50000 + np.cumsum(np.random.randn(1000) * 10),
        'volume': np.random.exponential(1000, 1000),
        'returns': np.random.randn(1000) * 0.01
    }, index=dates)
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Create baseline features
    baseline_df = fe.create_baseline_features(sample_data)
    print(f"Baseline features created: {baseline_df.shape}")
    
    # Create sample SAE features
    sae_processor = __import__('sae_processor').SAEFeatureProcessor()
    sae_df = sae_processor.create_sae_features(sample_data)
    
    # Integrate features
    combined_df = fe.integrate_sae_features(baseline_df, sae_df)
    print(f"Combined features: {combined_df.shape}")
    
    # Prepare for model
    features_df, feature_names = fe.prepare_features_for_model(combined_df, n_features=20)
    print(f"Final features for model: {features_df.shape}")
    print(f"Selected features: {feature_names[:10]}")

if __name__ == "__main__":
    main()
