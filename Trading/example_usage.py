"""
Example Usage of SAE-Based Trading System
Demonstrates how to use the trading pipeline with different configurations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Import our modules
from main_trading_script import TradingPipeline
from data_fetcher import CryptoDataFetcher
from sae_processor import SAEFeatureProcessor
from feature_engineer import FeatureEngineer
from prediction_model import TradingPredictionModel
from backtesting_engine import BacktestingEngine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_basic_usage():
    """Example 1: Basic usage with default configuration"""
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 60)
    
    # Initialize pipeline with default config
    pipeline = TradingPipeline()
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline()
    
    # Print results summary
    if 'backtest' in results:
        print("\nBacktest Results:")
        for strategy_name, strategy_results in results['backtest'].items():
            if 'metrics' in strategy_results:
                metrics = strategy_results['metrics']
                print(f"\n{strategy_name.upper()}:")
                print(f"  Total Return: {metrics['total_return']:.2%}")
                print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
                print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
                print(f"  Win Rate: {metrics['win_rate']:.2%}")
                print(f"  Number of Trades: {metrics['n_trades']:.0f}")

def example_custom_configuration():
    """Example 2: Custom configuration for different trading setup"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Custom Configuration")
    print("=" * 60)
    
    # Custom configuration
    config = {
        'symbols': ['BTC/USDT', 'ETH/USDT'],
        'timeframe': '5m',  # 5-minute data
        'days_back': 14,    # 2 weeks of data
        'model_type': 'xgboost',  # Single model instead of ensemble
        'confidence_threshold': 0.7,  # Higher confidence threshold
        'initial_cash': 50000,
        'fees': 0.001,  # 10 bps fees
        'slippage': 0.0005,
        'min_hold_period': 10,
        'max_hold_period': 120,
        'output_dir': 'results_custom'
    }
    
    # Initialize pipeline with custom config
    pipeline = TradingPipeline(config)
    
    # Run pipeline
    results = pipeline.run_complete_pipeline()
    
    # Print custom results
    if 'backtest' in results:
        print("\nCustom Configuration Results:")
        for strategy_name, strategy_results in results['backtest'].items():
            if 'metrics' in strategy_results:
                metrics = strategy_results['metrics']
                print(f"\n{strategy_name.upper()}:")
                print(f"  Total Return: {metrics['total_return']:.2%}")
                print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")

def example_individual_components():
    """Example 3: Using individual components separately"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Individual Components")
    print("=" * 60)
    
    # Step 1: Fetch data
    print("Step 1: Fetching data...")
    fetcher = CryptoDataFetcher()
    market_data = fetcher.fetch_ohlcv('BTC/USDT', '1m', since=datetime.now() - timedelta(days=7))
    print(f"Fetched {len(market_data)} candles")
    
    # Step 2: Process SAE features
    print("Step 2: Processing SAE features...")
    sae_processor = SAEFeatureProcessor()
    sae_data = sae_processor.create_sae_features(market_data)
    print(f"Created {len([col for col in sae_data.columns if col.startswith('SAE_')])} SAE features")
    
    # Step 3: Engineer features
    print("Step 3: Engineering features...")
    feature_engineer = FeatureEngineer()
    baseline_data = feature_engineer.create_baseline_features(market_data)
    combined_data = feature_engineer.integrate_sae_features(baseline_data, sae_data)
    print(f"Created {len(combined_data.columns)} total features")
    
    # Step 4: Train model
    print("Step 4: Training model...")
    model = TradingPredictionModel(model_type='random_forest')
    combined_data = model.create_target_variable(combined_data)
    
    # Get feature columns
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'returns', 'log_returns', 
                   'target', 'target_3class', 'target_continuous', 'target_vol_adj']
    feature_cols = [col for col in combined_data.columns if col not in exclude_cols]
    
    # Prepare data
    data_splits = model.prepare_training_data(combined_data, feature_cols)
    
    # Train model
    training_results = model.train_models(data_splits)
    print(f"Model training completed")
    
    # Step 5: Create signals
    print("Step 5: Creating signals...")
    predictions = model.predict(data_splits['X_test'])
    
    # Create signals
    backtesting_engine = BacktestingEngine()
    signals = backtesting_engine.create_trading_signals(
        combined_data.loc[data_splits['X_test'].index],
        predictions['predictions'],
        predictions['probabilities']
    )
    print(f"Created {signals['entry_signal'].sum()} entry signals")

def example_walk_forward_analysis():
    """Example 4: Walk-forward analysis"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Walk-Forward Analysis")
    print("=" * 60)
    
    # Configuration for walk-forward
    config = {
        'symbols': ['BTC/USDT'],
        'timeframe': '5m',
        'days_back': 60,  # More data for walk-forward
        'model_type': 'ensemble',
        'output_dir': 'results_wf'
    }
    
    pipeline = TradingPipeline(config)
    
    # Run walk-forward analysis
    wf_results = pipeline.run_walk_forward_analysis()
    
    if wf_results:
        print(f"\nWalk-Forward Analysis Results:")
        print(f"  Number of Periods: {wf_results['n_periods']}")
        print(f"  Average Return: {wf_results['avg_return']:.2%}")
        print(f"  Return Std Dev: {wf_results['std_return']:.2%}")
        print(f"  Average Sharpe: {wf_results['avg_sharpe']:.3f}")
        print(f"  Sharpe Std Dev: {wf_results['std_sharpe']:.3f}")
        print(f"  Positive Periods: {wf_results['positive_periods']}/{wf_results['n_periods']}")
        print(f"  Max Drawdown: {wf_results['max_drawdown']:.2%}")

def example_feature_analysis():
    """Example 5: Analyze SAE feature importance"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: SAE Feature Analysis")
    print("=" * 60)
    
    # Initialize SAE processor
    sae_processor = SAEFeatureProcessor()
    
    # Get feature importance
    importance_df = sae_processor.get_feature_importance()
    
    print("Top 10 Most Important SAE Features:")
    print(importance_df.head(10).to_string(index=False))
    
    # Create sample data and analyze features
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    sample_data = pd.DataFrame({
        'close': 50000 + np.cumsum(np.random.randn(1000) * 10),
        'volume': np.random.exponential(1000, 1000),
        'returns': np.random.randn(1000) * 0.01
    }, index=dates)
    
    # Create SAE features
    sae_data = sae_processor.create_sae_features(sample_data)
    
    # Analyze feature correlations
    sae_features = [col for col in sae_data.columns if col.startswith('SAE_')]
    correlations = sae_data[sae_features].corr()
    
    print(f"\nSAE Feature Statistics:")
    print(f"  Number of SAE Features: {len(sae_features)}")
    print(f"  Average Correlation: {correlations.values[np.triu_indices_from(correlations.values, k=1)].mean():.3f}")
    print(f"  Max Correlation: {correlations.values[np.triu_indices_from(correlations.values, k=1)].max():.3f}")

def example_risk_analysis():
    """Example 6: Risk analysis and metrics"""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Risk Analysis")
    print("=" * 60)
    
    # Create sample portfolio data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    
    # Simulate portfolio returns
    returns = np.random.normal(0.001, 0.02, 1000)  # 0.1% daily return, 2% volatility
    portfolio_value = 100000 * np.cumprod(1 + returns)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'close': portfolio_value,
        'volume': np.random.exponential(1000, 1000),
        'returns': returns
    }, index=dates)
    
    # Create sample signals
    signals = np.random.choice([0, 1], size=1000, p=[0.8, 0.2])
    signals_df = pd.DataFrame({
        'entry_signal': signals,
        'exit_signal': np.roll(signals, 10),
        'predictions': signals,
        'probabilities': np.random.uniform(0.5, 1.0, 1000),
        'position': signals
    }, index=dates)
    
    # Run backtest
    backtesting_engine = BacktestingEngine()
    results = backtesting_engine.run_backtest(sample_data, signals_df, 'risk_analysis')
    
    if results and 'metrics' in results:
        metrics = results['metrics']
        
        print("Risk Metrics:")
        print(f"  Total Return: {metrics['total_return']:.2%}")
        print(f"  Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"  Volatility: {metrics['volatility']:.2%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"  Sortino Ratio: {metrics['sortino_ratio']:.3f}")
        print(f"  Calmar Ratio: {metrics['calmar_ratio']:.3f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"  VaR (95%): {metrics['var_95']:.2%}")
        print(f"  VaR (99%): {metrics['var_99']:.2%}")
        print(f"  CVaR (95%): {metrics['cvar_95']:.2%}")
        print(f"  CVaR (99%): {metrics['cvar_99']:.2%}")

def main():
    """Run all examples"""
    print("SAE-Based Trading System - Examples")
    print("=" * 80)
    
    try:
        # Run examples
        example_basic_usage()
        example_custom_configuration()
        example_individual_components()
        example_walk_forward_analysis()
        example_feature_analysis()
        example_risk_analysis()
        
        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Error running examples: {str(e)}")
        print(f"\nError: {str(e)}")
        print("Please check your configuration and dependencies.")

if __name__ == "__main__":
    main()
