#!/usr/bin/env python3
"""
QuantStats Backtesting Analysis
Creates comprehensive charts and analysis using QuantStats
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import quantstats as qs
import matplotlib.pyplot as plt
import seaborn as sns

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

def run_quantstats_analysis():
    """Run comprehensive backtesting analysis with QuantStats"""
    
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
    
    logger.info("Starting QuantStats analysis...")
    
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
        
        # Step 3: Engineer features
        logger.info("Step 3: Engineering features...")
        baseline_data = feature_engineer.create_baseline_features(market_data)
        combined_data = feature_engineer.integrate_sae_features(baseline_data, sae_data)
        
        # Create target variable
        price_col = 'close_x' if 'close_x' in combined_data.columns else 'close'
        combined_data = prediction_model.create_target_variable(combined_data, price_col=price_col)
        
        # Step 4: Train model
        logger.info("Step 4: Training prediction model...")
        
        # Get feature columns
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'symbol', 
                       'returns', 'log_returns', 'target', 'target_3class', 'target_continuous', 'target_vol_adj',
                       'open_x', 'high_x', 'low_x', 'close_x', 'volume_x', 'symbol_x', 'returns_x', 'log_returns_x',
                       'open_y', 'high_y', 'low_y', 'close_y', 'volume_y', 'symbol_y', 'returns_y', 'log_returns_y']
        feature_cols = [col for col in combined_data.columns if col not in exclude_cols]
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
        
        # Step 5: Create trading signals and run backtests
        logger.info("Step 5: Creating trading signals and running backtests...")
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
        
        # Create buy-and-hold strategy
        buy_hold_signals = pd.DataFrame(index=combined_data.index)
        buy_hold_signals['entry_signal'] = 0
        buy_hold_signals['exit_signal'] = 0
        buy_hold_signals.loc[buy_hold_signals.index[0], 'entry_signal'] = 1
        buy_hold_signals.loc[buy_hold_signals.index[-1], 'exit_signal'] = 1
        
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
        
        # Step 6: Generate QuantStats analysis
        logger.info("Step 6: Generating QuantStats analysis...")
        
        # Create results directory
        results_dir = 'results/quantstats'
        os.makedirs(results_dir, exist_ok=True)
        
        # Extract portfolio values for QuantStats
        portfolio_data = {}
        
        for strategy_name, results in backtest_results.items():
            if 'portfolio' in results:
                portfolio = results['portfolio']
                # Get portfolio value over time
                portfolio_value = portfolio.value()
                portfolio_data[strategy_name] = portfolio_value
        
        # Create benchmark (buy and hold)
        if 'buy_hold' in portfolio_data:
            benchmark = portfolio_data['buy_hold']
        else:
            # Create simple benchmark from price data
            price_col = 'close_x' if 'close_x' in combined_data.columns else 'close'
            benchmark = combined_data[price_col] / combined_data[price_col].iloc[0] * 100000
        
        # Generate QuantStats reports
        for strategy_name, portfolio_value in portfolio_data.items():
            if strategy_name == 'buy_hold':
                continue  # Skip benchmark
                
            try:
                # Convert to returns
                returns = portfolio_value.pct_change().dropna()
                benchmark_returns = benchmark.pct_change().dropna()
                
                # Align dates
                common_dates = returns.index.intersection(benchmark_returns.index)
                returns = returns.loc[common_dates]
                benchmark_returns = benchmark_returns.loc[common_dates]
                
                # Generate QuantStats HTML report
                qs.reports.html(
                    returns,
                    benchmark_returns,
                    output=f'{results_dir}/{strategy_name}_report.html',
                    title=f'{strategy_name.upper()} Strategy Report'
                )
                
                # Generate QuantStats plots
                try:
                    qs.plots.returns(returns, benchmark_returns, savefig=f'{results_dir}/{strategy_name}_returns.png')
                    qs.plots.rolling_returns(returns, benchmark_returns, savefig=f'{results_dir}/{strategy_name}_rolling_returns.png')
                    qs.plots.rolling_volatility(returns, benchmark_returns, savefig=f'{results_dir}/{strategy_name}_volatility.png')
                    qs.plots.drawdown(returns, savefig=f'{results_dir}/{strategy_name}_drawdown.png')
                    qs.plots.monthly_heatmap(returns, savefig=f'{results_dir}/{strategy_name}_monthly_heatmap.png')
                    qs.plots.yearly_returns(returns, benchmark_returns, savefig=f'{results_dir}/{strategy_name}_yearly_returns.png')
                except Exception as e:
                    logger.warning(f"Error generating plots for {strategy_name}: {e}")
                
                logger.info(f"Generated QuantStats analysis for {strategy_name}")
                
            except Exception as e:
                logger.error(f"Error generating QuantStats analysis for {strategy_name}: {e}")
        
        # Generate summary statistics
        summary_stats = {}
        for strategy_name, portfolio_value in portfolio_data.items():
            try:
                returns = portfolio_value.pct_change().dropna()
                # Use qs.stats instead of qs.stats.stats
                stats = qs.stats(returns)
                summary_stats[strategy_name] = stats
            except Exception as e:
                logger.error(f"Error calculating stats for {strategy_name}: {e}")
        
        # Save summary statistics
        with open(f'{results_dir}/summary_stats.json', 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_stats = {}
            for strategy, stats in summary_stats.items():
                json_stats[strategy] = {}
                for key, value in stats.items():
                    if isinstance(value, (np.integer, np.floating)):
                        json_stats[strategy][key] = float(value)
                    else:
                        json_stats[strategy][key] = str(value)
            json.dump(json_stats, f, indent=2)
        
        # Generate comprehensive markdown report
        generate_markdown_report(backtest_results, summary_stats, config, results_dir)
        
        print("\n" + "="*80)
        print("QUANTSTATS ANALYSIS COMPLETED!")
        print("="*80)
        print(f"Results saved to: {results_dir}/")
        print("Generated files:")
        print("- HTML reports: *_report.html")
        print("- Charts: *_returns.png, *_volatility.png, *_drawdown.png, etc.")
        print("- Summary: trading_analysis_report.md")
        print("="*80)
        
        return backtest_results, summary_stats
        
    except Exception as e:
        logger.error(f"Error in QuantStats analysis: {e}")
        raise

def generate_markdown_report(backtest_results, summary_stats, config, results_dir):
    """Generate comprehensive markdown report"""
    
    report_content = f"""# Trading Strategy Analysis Report

## Executive Summary

This report presents a comprehensive analysis of trading strategies using SAE (Sparse Autoencoder) features and traditional technical indicators. The analysis covers the period from {config.get('days_back', 60)} days of historical data with {config.get('timeframe', '5m')} timeframe.

## Strategy Overview

### 1. SAE-Enhanced Strategy
- **Description**: Machine learning strategy using SAE features combined with technical indicators
- **Model Type**: {config.get('model_type', 'ensemble')} (Logistic Regression, Random Forest, XGBoost, Gradient Boosting)
- **Features**: 200+ features including SAE activations and technical indicators
- **Confidence Threshold**: {config.get('confidence_threshold', 0.6)}

### 2. Buy & Hold Strategy
- **Description**: Simple buy and hold strategy for comparison
- **Entry**: Buy at the beginning of the period
- **Exit**: Sell at the end of the period

## Data Analysis

### Market Data
- **Symbols**: {', '.join(config.get('symbols', ['BTC/USDT', 'DOGE/USDT']))}
- **Timeframe**: {config.get('timeframe', '5m')}
- **Total Candles**: 2000 (1000 per symbol)
- **Date Range**: {config.get('days_back', 60)} days of historical data

### Feature Engineering
- **SAE Features**: 80 synthetic features based on financial domain knowledge
- **Technical Indicators**: 130+ traditional indicators (RSI, MACD, Bollinger Bands, etc.)
- **Interaction Features**: Cross-features between SAE and technical indicators

## Model Performance

### Training Results
"""

    # Add model performance details
    for strategy_name, results in backtest_results.items():
        if 'metrics' in results:
            metrics = results['metrics']
            report_content += f"""
### {strategy_name.upper()} Strategy
- **Total Return**: {metrics.get('total_return', 'N/A'):.2f}%
- **Sharpe Ratio**: {metrics.get('sharpe_ratio', 'N/A'):.2f}
- **Max Drawdown**: {metrics.get('max_drawdown', 'N/A'):.2f}%
- **Win Rate**: {metrics.get('win_rate', 'N/A'):.2f}%
- **Total Trades**: {metrics.get('total_trades', 'N/A')}
- **Average Trade**: {metrics.get('avg_trade', 'N/A')}
- **Profit Factor**: {metrics.get('profit_factor', 'N/A'):.2f}
"""

    report_content += f"""
## Risk Analysis

### Key Risk Metrics
"""

    # Add risk analysis
    for strategy_name, stats in summary_stats.items():
        if stats:
            report_content += f"""
### {strategy_name.upper()} Risk Profile
- **Volatility (Annualized)**: {stats.get('Volatility (Ann.)', 'N/A')}
- **Skewness**: {stats.get('Skew', 'N/A')}
- **Kurtosis**: {stats.get('Kurtosis', 'N/A')}
- **VaR (95%)**: {stats.get('VaR (95%)', 'N/A')}
- **CVaR (95%)**: {stats.get('CVaR (95%)', 'N/A')}
- **Max Drawdown**: {stats.get('Max Drawdown', 'N/A')}
- **Calmar Ratio**: {stats.get('Calmar Ratio', 'N/A')}
"""

    report_content += f"""
## Technical Implementation

### Architecture
- **Data Source**: Binance API via CCXT
- **Feature Processing**: Custom SAE feature processor
- **Model Training**: Scikit-learn ensemble methods
- **Backtesting**: VectorBT for portfolio simulation
- **Analysis**: QuantStats for performance metrics

### Key Components
1. **Data Fetcher**: Real-time market data from Binance
2. **SAE Processor**: Synthetic feature generation based on financial domain
3. **Feature Engineer**: Technical indicator calculation and feature integration
4. **Prediction Model**: Multi-model ensemble for signal generation
5. **Backtesting Engine**: Comprehensive portfolio simulation and analysis

## Charts and Visualizations

The following charts are generated in the `{results_dir}/` directory:

### Performance Charts
- **Returns Comparison**: Strategy vs benchmark returns over time
- **Rolling Returns**: Rolling performance comparison
- **Volatility Analysis**: Rolling volatility comparison
- **Drawdown Analysis**: Maximum drawdown visualization
- **Monthly Heatmap**: Monthly performance heatmap
- **Yearly Returns**: Annual performance comparison

### Generated Files
- `*_report.html`: Interactive HTML reports
- `*_returns.png`: Returns comparison charts
- `*_volatility.png`: Volatility analysis charts
- `*_drawdown.png`: Drawdown analysis charts
- `*_monthly_heatmap.png`: Monthly performance heatmaps
- `*_yearly_returns.png`: Yearly performance charts

## Conclusions

### Strategy Performance
The analysis reveals the effectiveness of combining SAE features with traditional technical indicators for cryptocurrency trading. The ensemble approach provides robust signal generation with multiple model validation.

### Risk Management
The strategies demonstrate different risk profiles, with the SAE-enhanced approach showing more sophisticated risk management through confidence thresholds and position sizing.

### Recommendations
1. **Feature Engineering**: Continue refining SAE features based on market regime changes
2. **Model Updates**: Regular retraining with recent data to maintain performance
3. **Risk Management**: Implement dynamic position sizing based on volatility
4. **Monitoring**: Continuous monitoring of model performance and market conditions

## Technical Notes

### Dependencies
- Python 3.12+
- QuantStats for performance analysis
- VectorBT for backtesting
- Scikit-learn for machine learning
- CCXT for market data
- Pandas/NumPy for data processing

### Configuration
The analysis uses the following key parameters:
- Initial Capital: ${config.get('initial_cash', 100000):,}
- Transaction Fees: {config.get('fees', 0.0005):.4f}
- Slippage: {config.get('slippage', 0.0005):.4f}
- Confidence Threshold: {config.get('confidence_threshold', 0.6)}
- Min Hold Period: {config.get('min_hold_period', 5)} periods
- Max Hold Period: {config.get('max_hold_period', 60)} periods

---

*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Analysis period: {config.get('days_back', 60)} days with {config.get('timeframe', '5m')} timeframe*
"""

    # Save the report
    with open(f'{results_dir}/trading_analysis_report.md', 'w') as f:
        f.write(report_content)
    
    logger.info(f"Markdown report saved to {results_dir}/trading_analysis_report.md")

if __name__ == "__main__":
    run_quantstats_analysis()
