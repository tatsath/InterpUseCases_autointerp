#!/usr/bin/env python3
"""
Run Backtesting Only
Focuses on just the backtesting functionality
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

# Import our custom modules
from data_fetcher import CryptoDataFetcher
from sae_processor import SAEFeatureProcessor
from feature_engineer import FeatureEngineer
from prediction_model import TradingPredictionModel
from backtesting_engine import BacktestingEngine

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_backtest_only():
    """Run just the backtesting functionality"""
    
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Override with command line arguments
    config.update({
        'symbols': ['BTC/USDT', 'DOGE/USDT'],
        'timeframe': '5m',
        'days_back': 60,
        'model_type': 'ensemble',
        'output_dir': 'results'
    })
    
    logger.info("Starting backtesting pipeline...")
    
    # Initialize components
    data_fetcher = CryptoDataFetcher()
    sae_processor = SAEFeatureProcessor()
    feature_engineer = FeatureEngineer()
    prediction_model = TradingPredictionModel(model_type=config['model_type'])
    backtesting_engine = BacktestingEngine(
        initial_cash=config['initial_cash'],
        fees=config['fees'],
        slippage=config['slippage']
    )
    
    try:
        # Step 1: Fetch market data
        logger.info("Step 1: Fetching market data...")
        market_data = data_fetcher.fetch_multiple_symbols(
            config['symbols'], 
            config['timeframe']
        )
        logger.info(f"Fetched {len(market_data)} candles for {len(config['symbols'])} symbols")
        
        # Step 2: Process SAE features
        logger.info("Step 2: Processing SAE features...")
        sae_data = sae_processor.create_sae_features(market_data)
        sae_data.to_parquet('results/sae_features.parquet')
        logger.info("SAE features saved to results/sae_features.parquet")
        
        # Step 3: Engineer features
        logger.info("Step 3: Engineering features...")
        baseline_data = feature_engineer.create_baseline_features(market_data)
        combined_data = feature_engineer.integrate_sae_features(baseline_data, sae_data)
        
        # Create target variable (use close_x as the price column after merge)
        price_col = 'close_x' if 'close_x' in combined_data.columns else 'close'
        combined_data = prediction_model.create_target_variable(combined_data, price_col=price_col)
        
        # Step 4: Train model (simplified)
        logger.info("Step 4: Training prediction model...")
        
        # Get feature columns (exclude all non-numeric columns)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'symbol', 
                       'returns', 'log_returns', 'target', 'target_3class', 'target_continuous', 'target_vol_adj',
                       'open_x', 'high_x', 'low_x', 'close_x', 'volume_x', 'symbol_x', 'returns_x', 'log_returns_x',
                       'open_y', 'high_y', 'low_y', 'close_y', 'volume_y', 'symbol_y', 'returns_y', 'log_returns_y']
        feature_cols = [col for col in combined_data.columns if col not in exclude_cols]
        
        # Additional filter to ensure only numeric columns are included
        numeric_cols = combined_data[feature_cols].select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in feature_cols if col in numeric_cols]
        
        # Prepare training data
        data_splits = prediction_model.prepare_training_data(
            combined_data, 
            feature_cols,
            test_size=config['test_size'],
            validation_size=config['validation_size']
        )
        
        # Train models
        training_results = prediction_model.train_models(data_splits)
        evaluation_results = prediction_model.evaluate_models(data_splits)
        
        # Step 5: Create trading signals
        logger.info("Step 5: Creating trading signals...")
        test_data = data_splits['X_test']
        
        # Make predictions
        predictions = prediction_model.predict(test_data, model_name='ensemble')
        
        # Create signals DataFrame
        signals_df = pd.DataFrame(index=test_data.index)
        signals_df['predictions'] = predictions['predictions']
        signals_df['probabilities'] = predictions['probabilities'] if predictions['probabilities'] is not None else np.ones_like(predictions['predictions']) * 0.5
        
        # Create trading signals
        signals_df = backtesting_engine.create_trading_signals(
            combined_data.loc[test_data.index],
            signals_df['predictions'].values,
            signals_df['probabilities'].values,
            confidence_threshold=config['confidence_threshold'],
            min_hold_period=config['min_hold_period'],
            max_hold_period=config['max_hold_period']
        )
        
        # Step 6: Run backtesting
        logger.info("Step 6: Running backtesting...")
        
        # Create simple buy-and-hold strategy for comparison
        buy_hold_signals = pd.DataFrame(index=combined_data.index)
        buy_hold_signals['entry_signal'] = 0
        buy_hold_signals['exit_signal'] = 0
        buy_hold_signals.loc[buy_hold_signals.index[0], 'entry_signal'] = 1  # Buy at start
        buy_hold_signals.loc[buy_hold_signals.index[-1], 'exit_signal'] = 1  # Sell at end
        
        # Run backtests
        backtest_results = {}
        
        # SAE Strategy
        try:
            sae_backtest = backtesting_engine.run_backtest(
                combined_data, 
                signals_df,
                strategy_name='sae_strategy'
            )
            backtest_results['sae_strategy'] = sae_backtest
            logger.info("SAE strategy backtest completed")
        except Exception as e:
            logger.error(f"Error in SAE strategy backtest: {e}")
        
        # Buy and Hold Strategy
        try:
            buy_hold_backtest = backtesting_engine.run_backtest(
                combined_data, 
                buy_hold_signals,
                strategy_name='buy_hold'
            )
            backtest_results['buy_hold'] = buy_hold_backtest
            logger.info("Buy and hold strategy backtest completed")
        except Exception as e:
            logger.error(f"Error in buy and hold backtest: {e}")
        
        # Print results
        print("\n" + "="*80)
        print("BACKTESTING RESULTS")
        print("="*80)
        
        for strategy_name, results in backtest_results.items():
            if 'metrics' in results:
                metrics = results['metrics']
                print(f"\n{strategy_name.upper()}:")
                print(f"  Total Return: {metrics.get('total_return', 'N/A'):.2f}%")
                print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A'):.2f}")
                print(f"  Max Drawdown: {metrics.get('max_drawdown', 'N/A'):.2f}%")
                print(f"  Win Rate: {metrics.get('win_rate', 'N/A'):.2f}%")
                print(f"  Total Trades: {metrics.get('total_trades', 'N/A')}")
        
        print("\n" + "="*80)
        print("BACKTESTING COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        return backtest_results
        
    except Exception as e:
        logger.error(f"Error in backtesting pipeline: {e}")
        raise

if __name__ == "__main__":
    run_backtest_only()

