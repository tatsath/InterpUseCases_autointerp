"""
Demo Trading System with Synthetic Data
Demonstrates the complete SAE-based trading pipeline using synthetic market data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os

# Import our modules
from sae_processor import SAEFeatureProcessor
from feature_engineer import FeatureEngineer
from prediction_model import TradingPredictionModel
from backtesting_engine import BacktestingEngine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DemoTradingSystem:
    """Demo trading system using synthetic data"""
    
    def __init__(self):
        """Initialize the demo system"""
        self.sae_processor = SAEFeatureProcessor()
        self.feature_engineer = FeatureEngineer()
        self.prediction_model = TradingPredictionModel(model_type='ensemble')
        self.backtesting_engine = BacktestingEngine()
        
    def create_synthetic_market_data(self, 
                                   symbols: list = ['BTC/USDT', 'DOGE/USDT'],
                                   days: int = 30,
                                   timeframe: str = '1min') -> pd.DataFrame:
        """
        Create realistic synthetic market data
        
        Args:
            symbols: List of trading symbols
            days: Number of days of data
            timeframe: Data frequency
            
        Returns:
            DataFrame with synthetic OHLCV data
        """
        # Calculate number of periods based on timeframe
        if timeframe == '1min':
            periods = days * 24 * 60
        elif timeframe == '5min':
            periods = days * 24 * 12
        elif timeframe == '1h':
            periods = days * 24
        else:
            periods = days * 24 * 60  # Default to 1min
        
        all_data = []
        
        for symbol in symbols:
            logger.info(f"Creating synthetic data for {symbol}...")
            
            # Create realistic price movements
            np.random.seed(hash(symbol) % 2**32)  # Different seed per symbol
            
            # Base price
            if 'BTC' in symbol:
                base_price = 50000
                volatility = 0.02
            elif 'DOGE' in symbol:
                base_price = 0.08
                volatility = 0.05
            else:
                base_price = 100
                volatility = 0.03
            
            # Generate price series with trend and volatility
            returns = np.random.normal(0.0001, volatility, periods)  # Slight upward trend
            
            # Add some market cycles
            cycle_period = periods // 7  # Weekly cycles
            cycle_effect = 0.001 * np.sin(2 * np.pi * np.arange(periods) / cycle_period)
            returns += cycle_effect
            
            # Add some volatility clustering
            vol_cluster = np.random.exponential(1, periods)
            returns *= vol_cluster
            
            # Calculate prices
            prices = base_price * np.exp(np.cumsum(returns))
            
            # Create OHLCV data
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=days),
                periods=periods,
                freq=timeframe
            )
            
            # Generate realistic OHLC from close prices
            high_multiplier = 1 + np.random.exponential(0.005, periods)
            low_multiplier = 1 - np.random.exponential(0.005, periods)
            
            df = pd.DataFrame({
                'open': prices * (1 + np.random.normal(0, 0.001, periods)),
                'high': prices * high_multiplier,
                'low': prices * low_multiplier,
                'close': prices,
                'volume': np.random.exponential(1000, periods) * (1 + np.abs(returns) * 10)
            }, index=dates)
            
            # Ensure high >= max(open, close) and low <= min(open, close)
            df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
            df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
            
            # Add returns and log returns
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['symbol'] = symbol.replace('/', '')
            
            all_data.append(df)
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=False)
        combined_data = combined_data.sort_index()
        
        logger.info(f"Created synthetic data: {len(combined_data)} candles for {len(symbols)} symbols")
        return combined_data
    
    def run_complete_demo(self) -> dict:
        """Run the complete demo trading pipeline"""
        logger.info("Starting demo trading pipeline...")
        
        try:
            # Step 1: Create synthetic market data
            logger.info("Step 1: Creating synthetic market data...")
            market_data = self.create_synthetic_market_data(
                symbols=['BTC/USDT', 'DOGE/USDT'],
                days=30,
                timeframe='1min'
            )
            
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
            
            # Step 5: Prepare training data
            logger.info("Step 5: Preparing training data...")
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'symbol', 
                           'returns', 'log_returns', 'target', 'target_3class', 'target_continuous', 'target_vol_adj']
            feature_cols = [col for col in combined_data.columns if col not in exclude_cols]
            
            data_splits = self.prediction_model.prepare_training_data(
                combined_data, 
                feature_cols,
                test_size=0.2,
                validation_size=0.1
            )
            
            # Step 6: Train models
            logger.info("Step 6: Training prediction models...")
            training_results = self.prediction_model.train_models(data_splits)
            
            # Step 7: Evaluate models
            logger.info("Step 7: Evaluating models...")
            evaluation_results = self.prediction_model.evaluate_models(data_splits)
            
            # Step 8: Create trading signals
            logger.info("Step 8: Creating trading signals...")
            predictions = self.prediction_model.predict(data_splits['X_test'], model_name='ensemble')
            
            # Create signals for each symbol
            all_signals = []
            for symbol in combined_data['symbol'].unique():
                symbol_data = combined_data[combined_data['symbol'] == symbol]
                symbol_test_data = data_splits['X_test'][data_splits['X_test'].index.isin(symbol_data.index)]
                
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
                signals_combined = pd.concat(all_signals, ignore_index=False)
            else:
                logger.warning("No signals created")
                return {}
            
            # Step 9: Run backtesting
            logger.info("Step 9: Running backtesting...")
            backtest_results = {}
            
            for symbol in combined_data['symbol'].unique():
                symbol_data = market_data[market_data['symbol'] == symbol]
                symbol_signals = signals_combined[signals_combined.index.isin(symbol_data.index)]
                
                if len(symbol_signals) > 0:
                    symbol_results = self.backtesting_engine.run_backtest(
                        symbol_data, 
                        symbol_signals, 
                        f'sae_strategy_{symbol}'
                    )
                    if symbol_results:
                        backtest_results[f'sae_strategy_{symbol}'] = symbol_results
            
            # Create baseline comparison
            logger.info("Step 10: Creating baseline comparison...")
            baseline_results = {}
            
            for symbol in combined_data['symbol'].unique():
                symbol_data = market_data[market_data['symbol'] == symbol]
                
                # Simple EMA crossover baseline
                ema_fast = symbol_data['close'].ewm(span=20).mean()
                ema_slow = symbol_data['close'].ewm(span=50).mean()
                
                baseline_signals = pd.DataFrame(index=symbol_data.index)
                baseline_signals['predictions'] = (ema_fast > ema_slow).astype(int)
                baseline_signals['probabilities'] = np.where(baseline_signals['predictions'] == 1, 0.7, 0.3)
                
                baseline_signals = self.backtesting_engine.create_trading_signals(
                    symbol_data,
                    baseline_signals['predictions'].values,
                    baseline_signals['probabilities'].values,
                    confidence_threshold=0.5,
                    min_hold_period=10,
                    max_hold_period=120
                )
                
                baseline_result = self.backtesting_engine.run_backtest(
                    symbol_data,
                    baseline_signals,
                    f'baseline_strategy_{symbol}'
                )
                
                if baseline_result:
                    baseline_results[f'baseline_strategy_{symbol}'] = baseline_result
            
            # Combine all results
            all_results = {**backtest_results, **baseline_results}
            
            # Step 11: Generate reports
            logger.info("Step 11: Generating reports...")
            self._generate_demo_reports(all_results, training_results, evaluation_results)
            
            logger.info("Demo trading pipeline completed successfully!")
            return {
                'market_data': market_data,
                'sae_data': sae_data,
                'combined_data': combined_data,
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'backtest_results': all_results
            }
            
        except Exception as e:
            logger.error(f"Error in demo pipeline: {str(e)}")
            raise
    
    def _generate_demo_reports(self, backtest_results, training_results, evaluation_results):
        """Generate demo reports"""
        os.makedirs('demo_results', exist_ok=True)
        
        # Generate text report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("SAE-BASED TRADING SYSTEM - DEMO RESULTS")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        report_lines.append("This demo uses synthetic market data to demonstrate the trading system.")
        report_lines.append("")
        
        # Model performance
        report_lines.append("MODEL PERFORMANCE")
        report_lines.append("-" * 40)
        for model_name, results in evaluation_results.items():
            if 'test_accuracy' in results:
                report_lines.append(f"{model_name.upper()}:")
                report_lines.append(f"  Test Accuracy: {results['test_accuracy']:.2%}")
                if results.get('test_auc'):
                    report_lines.append(f"  Test AUC: {results['test_auc']:.3f}")
                report_lines.append("")
        
        # Trading performance
        report_lines.append("TRADING PERFORMANCE")
        report_lines.append("-" * 40)
        for strategy_name, strategy_results in backtest_results.items():
            if 'metrics' in strategy_results:
                metrics = strategy_results['metrics']
                report_lines.append(f"{strategy_name.upper()}:")
                report_lines.append(f"  Total Return: {metrics['total_return']:.2%}")
                report_lines.append(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
                report_lines.append(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
                report_lines.append(f"  Win Rate: {metrics['win_rate']:.2%}")
                report_lines.append(f"  Number of Trades: {metrics['n_trades']:.0f}")
                report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        # Save report
        with open('demo_results/demo_report.txt', 'w') as f:
            f.write(report_text)
        
        # Print summary
        print("\n" + "=" * 80)
        print("DEMO TRADING SYSTEM RESULTS")
        print("=" * 80)
        print(report_text)
        
        logger.info("Demo reports generated in 'demo_results/' directory")

def main():
    """Run the demo trading system"""
    print("SAE-Based Trading System - Demo with Synthetic Data")
    print("=" * 80)
    print("This demo uses synthetic market data to demonstrate the complete pipeline.")
    print("For real trading, you would need to register for Binance API access.")
    print("=" * 80)
    
    try:
        # Initialize demo system
        demo = DemoTradingSystem()
        
        # Run complete demo
        results = demo.run_complete_demo()
        
        if results:
            print("\n" + "=" * 80)
            print("DEMO COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print("Check the 'demo_results/' directory for detailed reports.")
            print("\nTo use with real data:")
            print("1. Register for Binance API (free)")
            print("2. Update data_fetcher.py with your API credentials")
            print("3. Run main_trading_script.py")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"\nDemo failed: {str(e)}")

if __name__ == "__main__":
    main()
