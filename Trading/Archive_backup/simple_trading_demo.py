"""
Simple Trading Demo with Free Data Sources
Uses free APIs (Yahoo Finance, CoinGecko) - no authentication required
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os

# Import our modules
from free_data_fetcher import FreeCryptoDataFetcher
from sae_processor import SAEFeatureProcessor
from feature_engineer import FeatureEngineer
from prediction_model import TradingPredictionModel
from backtesting_engine import BacktestingEngine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleTradingDemo:
    """Simple trading demo using free data sources"""
    
    def __init__(self):
        """Initialize the demo system"""
        self.data_fetcher = FreeCryptoDataFetcher()
        self.sae_processor = SAEFeatureProcessor()
        self.feature_engineer = FeatureEngineer()
        self.prediction_model = TradingPredictionModel(model_type='ensemble')
        self.backtesting_engine = BacktestingEngine()
        
    def run_demo(self, use_real_data: bool = True):
        """
        Run the complete trading demo
        
        Args:
            use_real_data: If True, try to fetch real data; if False, use synthetic data
        """
        logger.info("Starting Simple Trading Demo...")
        
        try:
            # Step 1: Get market data
            logger.info("Step 1: Fetching market data...")
            
            if use_real_data:
                # Try to get real data first
                market_data = self._get_real_data()
                if market_data.empty:
                    logger.warning("Real data not available, using synthetic data...")
                    market_data = self._get_synthetic_data()
            else:
                market_data = self._get_synthetic_data()
            
            # Step 2: Process SAE features
            logger.info("Step 2: Processing SAE features...")
            sae_data = self.sae_processor.create_sae_features(market_data)
            
            # Step 3: Engineer features
            logger.info("Step 3: Engineering features...")
            baseline_data = self.feature_engineer.create_baseline_features(market_data)
            combined_data = self.feature_engineer.integrate_sae_features(baseline_data, sae_data)
            
            # Step 4: Create target variable
            logger.info("Step 4: Creating target variable...")
            combined_data = self.prediction_model.create_target_variable(combined_data)
            
            # Step 5: Train model
            logger.info("Step 5: Training prediction model...")
            model_results = self._train_model(combined_data)
            
            # Step 6: Create trading signals
            logger.info("Step 6: Creating trading signals...")
            signals = self._create_signals(combined_data, model_results)
            
            # Step 7: Run backtesting
            logger.info("Step 7: Running backtesting...")
            backtest_results = self._run_backtesting(market_data, signals)
            
            # Step 8: Generate reports
            logger.info("Step 8: Generating reports...")
            self._generate_reports(backtest_results, model_results)
            
            logger.info("Demo completed successfully!")
            return {
                'market_data': market_data,
                'sae_data': sae_data,
                'combined_data': combined_data,
                'model_results': model_results,
                'backtest_results': backtest_results
            }
            
        except Exception as e:
            logger.error(f"Demo failed: {str(e)}")
            raise
    
    def _get_real_data(self) -> pd.DataFrame:
        """Try to get real data from free sources"""
        try:
            # Try Yahoo Finance first
            logger.info("Trying Yahoo Finance...")
            data = self.data_fetcher.fetch_multiple_symbols(
                ['BTC', 'ETH', 'DOGE'],
                source='yahoo',
                period='30d',
                interval='1h'
            )
            
            if not data.empty:
                logger.info(f"Successfully fetched real data: {len(data)} candles")
                return data
            
            # Try CoinGecko as backup
            logger.info("Trying CoinGecko...")
            data = self.data_fetcher.fetch_multiple_symbols(
                ['BTC', 'ETH', 'DOGE'],
                source='coingecko',
                days=30
            )
            
            if not data.empty:
                logger.info(f"Successfully fetched real data: {len(data)} candles")
                return data
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.warning(f"Real data fetch failed: {str(e)}")
            return pd.DataFrame()
    
    def _get_synthetic_data(self) -> pd.DataFrame:
        """Get synthetic data for testing"""
        logger.info("Creating synthetic data...")
        data = self.data_fetcher.create_synthetic_data(
            symbols=['BTC', 'ETH', 'DOGE'],
            days=30,
            timeframe='1min'
        )
        logger.info(f"Created synthetic data: {len(data)} candles")
        return data
    
    def _train_model(self, data: pd.DataFrame) -> dict:
        """Train the prediction model"""
        # Get feature columns
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'symbol', 
                       'returns', 'log_returns', 'target', 'target_3class', 'target_continuous', 'target_vol_adj']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        # Prepare training data
        data_splits = self.prediction_model.prepare_training_data(
            data, 
            feature_cols,
            test_size=0.2,
            validation_size=0.1
        )
        
        # Train models
        training_results = self.prediction_model.train_models(data_splits)
        
        # Evaluate models
        evaluation_results = self.prediction_model.evaluate_models(data_splits)
        
        return {
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'data_splits': data_splits,
            'feature_cols': feature_cols
        }
    
    def _create_signals(self, data: pd.DataFrame, model_results: dict) -> pd.DataFrame:
        """Create trading signals"""
        test_data = model_results['data_splits']['X_test']
        
        # Make predictions
        predictions = self.prediction_model.predict(test_data, model_name='ensemble')
        
        # Create signals for each symbol
        all_signals = []
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol]
            symbol_test_data = test_data[test_data.index.isin(symbol_data.index)]
            
            if len(symbol_test_data) > 0:
                symbol_predictions = self.prediction_model.predict(symbol_test_data, model_name='ensemble')
                
                signals_df = self.backtesting_engine.create_trading_signals(
                    symbol_data.loc[symbol_test_data.index],
                    symbol_predictions['predictions'],
                    symbol_predictions['probabilities'],
                    confidence_threshold=0.6,
                    min_hold_period=5,
                    max_hold_period=60
                )
                all_signals.append(signals_df)
        
        if all_signals:
            return pd.concat(all_signals, ignore_index=False)
        else:
            return pd.DataFrame()
    
    def _run_backtesting(self, market_data: pd.DataFrame, signals: pd.DataFrame) -> dict:
        """Run backtesting"""
        backtest_results = {}
        
        for symbol in market_data['symbol'].unique():
            symbol_data = market_data[market_data['symbol'] == symbol]
            symbol_signals = signals[signals.index.isin(symbol_data.index)]
            
            if len(symbol_signals) > 0:
                # SAE strategy
                sae_result = self.backtesting_engine.run_backtest(
                    symbol_data, 
                    symbol_signals, 
                    f'sae_strategy_{symbol}'
                )
                if sae_result:
                    backtest_results[f'sae_strategy_{symbol}'] = sae_result
                
                # Baseline strategy
                baseline_signals = self._create_baseline_signals(symbol_data)
                baseline_result = self.backtesting_engine.run_backtest(
                    symbol_data,
                    baseline_signals,
                    f'baseline_strategy_{symbol}'
                )
                if baseline_result:
                    backtest_results[f'baseline_strategy_{symbol}'] = baseline_result
        
        return backtest_results
    
    def _create_baseline_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create baseline EMA crossover signals"""
        ema_fast = data['close'].ewm(span=20).mean()
        ema_slow = data['close'].ewm(span=50).mean()
        
        baseline_signals = pd.DataFrame(index=data.index)
        baseline_signals['predictions'] = (ema_fast > ema_slow).astype(int)
        baseline_signals['probabilities'] = np.where(baseline_signals['predictions'] == 1, 0.7, 0.3)
        
        return self.backtesting_engine.create_trading_signals(
            data,
            baseline_signals['predictions'].values,
            baseline_signals['probabilities'].values,
            confidence_threshold=0.5,
            min_hold_period=10,
            max_hold_period=120
        )
    
    def _generate_reports(self, backtest_results: dict, model_results: dict):
        """Generate demo reports"""
        os.makedirs('demo_results', exist_ok=True)
        
        # Print summary
        print("\n" + "=" * 80)
        print("SAE-BASED TRADING SYSTEM - DEMO RESULTS")
        print("=" * 80)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("")
        
        # Model performance
        print("MODEL PERFORMANCE")
        print("-" * 40)
        for model_name, results in model_results['evaluation_results'].items():
            if 'test_accuracy' in results:
                print(f"{model_name.upper()}:")
                print(f"  Test Accuracy: {results['test_accuracy']:.2%}")
                if results.get('test_auc'):
                    print(f"  Test AUC: {results['test_auc']:.3f}")
                print("")
        
        # Trading performance
        print("TRADING PERFORMANCE")
        print("-" * 40)
        for strategy_name, strategy_results in backtest_results.items():
            if 'metrics' in strategy_results:
                metrics = strategy_results['metrics']
                print(f"{strategy_name.upper()}:")
                print(f"  Total Return: {metrics['total_return']:.2%}")
                print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
                print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
                print(f"  Win Rate: {metrics['win_rate']:.2%}")
                print(f"  Number of Trades: {metrics['n_trades']:.0f}")
                print("")
        
        print("=" * 80)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)

def main():
    """Run the simple trading demo"""
    print("SAE-Based Trading System - Simple Demo")
    print("=" * 80)
    print("This demo uses FREE data sources (no authentication required):")
    print("- Yahoo Finance (yfinance)")
    print("- CoinGecko API")
    print("- Synthetic data (if real data fails)")
    print("=" * 80)
    
    try:
        # Initialize demo
        demo = SimpleTradingDemo()
        
        # Run demo with real data
        print("\nTrying to fetch real data...")
        results = demo.run_demo(use_real_data=True)
        
        if results:
            print("\n✅ Demo completed successfully!")
            print("Check the output above for trading performance results.")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        print("This might be due to network issues or API rate limits.")
        print("Try running again in a few minutes.")

if __name__ == "__main__":
    main()


