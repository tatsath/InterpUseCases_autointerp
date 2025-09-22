# SAE-Based Trading System

A comprehensive trading system that integrates Sparse Autoencoder (SAE) features from financial language models with traditional technical analysis for cryptocurrency trading.

## Overview

This system combines:
- **SAE Features**: Finance-specific features extracted from Llama-2-7B model trained on financial data
- **Technical Analysis**: Traditional momentum, volatility, and trend indicators
- **Machine Learning**: Ensemble models for price direction prediction
- **Backtesting**: Comprehensive backtesting with VectorBT
- **Risk Management**: Advanced risk metrics and walk-forward analysis

## Features

### SAE Feature Integration
- **50+ Finance Features**: Based on analysis from 5 layers (4, 10, 16, 22, 28)
- **Feature Weights**: Automatically weighted by F1 scores and specialization
- **Layer Aggregation**: Composite features across different model layers
- **Interaction Features**: Cross-features between SAE and technical indicators

### Technical Analysis
- **Price Features**: Ratios, gaps, true range, position within bars
- **Volume Features**: Volume ratios, momentum, on-balance volume
- **Momentum Indicators**: RSI, MACD, Stochastic, Williams %R
- **Volatility Indicators**: ATR, Bollinger Bands, volatility regimes
- **Trend Indicators**: ADX, Parabolic SAR, linear regression slopes

### Machine Learning Models
- **Ensemble Approach**: Logistic Regression, Random Forest, XGBoost, Gradient Boosting
- **Feature Selection**: Automatic feature selection based on importance
- **Time Series Validation**: Proper time series cross-validation
- **Confidence Thresholding**: Only trade when model is confident

### Backtesting Engine
- **VectorBT Integration**: Fast, vectorized backtesting
- **Multiple Strategies**: SAE-based, baseline, buy-and-hold comparison
- **Risk Metrics**: Sharpe, Sortino, Calmar ratios, VaR, CVaR
- **Walk-Forward Analysis**: Out-of-sample testing with rolling windows
- **Transaction Costs**: Realistic fees and slippage modeling

## Installation

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Verify Installation**:
```bash
python -c "import vectorbt, ccxt, sklearn, xgboost; print('All dependencies installed successfully!')"
```

## Quick Start

### Basic Usage

```bash
# Run with default configuration
python main_trading_script.py

# Run with custom parameters
python main_trading_script.py --symbols BTC/USDT DOGE/USDT --timeframe 5m --days 60

# Run with walk-forward analysis
python main_trading_script.py --walk-forward --output results_wf
```

### Configuration

Edit `config.json` to customize:
- Trading symbols and timeframes
- Model parameters and thresholds
- Backtesting settings
- Risk management rules

### Example Configuration

```json
{
  "symbols": ["BTC/USDT", "DOGE/USDT"],
  "timeframe": "1m",
  "days_back": 30,
  "model_type": "ensemble",
  "confidence_threshold": 0.6,
  "initial_cash": 100000,
  "fees": 0.0005,
  "slippage": 0.0005
}
```

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Fetcher  │───▶│  SAE Processor   │───▶│ Feature Engineer│
│   (CCXT)        │    │  (Finance SAEs)  │    │ (Technical + SAE)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Backtesting     │◀───│ Prediction Model │◀───│                 │
│ Engine (VectorBT)│    │ (Ensemble ML)    │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## SAE Features

### Layer 4 Features (Early Processing)
- **Earnings Reports** (F1: 0.72): High-performance feature for earnings-related patterns
- **Stock Index Performance** (F1: 0.84): Strong performance for market trends
- **Inflation Indicators** (F1: 0.84): Economic indicator sensitivity

### Layer 10 Features (Mid Processing)
- **Revenue Metrics** (F1: 0.9): Highest performing feature for financial metrics
- **Stock Performance** (F1: 0.865): Strong market trend detection
- **Cryptocurrency** (F1: 0.673): Crypto-specific patterns

### Layer 16 Features (Advanced Processing)
- **Earnings Reports** (F1: 0.769): Specialized earnings analysis
- **Major Figures** (F1: 0.692): Key financial figure recognition
- **Central Bank Policies** (F1: 0.76): Policy impact detection

### Layer 22 Features (High-Level Processing)
- **Performance Updates** (F1: 0.808): Performance tracking
- **Trading Strategies** (F1: 0.712): Strategy pattern recognition
- **Fintech Solutions** (F1: 0.76): Technology sector patterns

### Layer 28 Features (Deep Processing)
- **Portfolio Diversification** (F1: 0.865): Diversification strategies
- **Sector Investment** (F1: 0.788): Sector-specific patterns
- **Revenue Figures** (F1: 0.808): Financial performance metrics

## Usage Examples

### 1. Basic Trading Pipeline

```python
from main_trading_script import TradingPipeline

# Initialize with default config
pipeline = TradingPipeline()

# Run complete pipeline
results = pipeline.run_complete_pipeline()

# Access results
print(f"SAE Strategy Return: {results['backtest']['sae_strategy']['metrics']['total_return']:.2%}")
```

### 2. Custom Configuration

```python
config = {
    'symbols': ['BTC/USDT', 'ETH/USDT'],
    'timeframe': '5m',
    'days_back': 60,
    'confidence_threshold': 0.7,
    'model_type': 'xgboost'
}

pipeline = TradingPipeline(config)
results = pipeline.run_complete_pipeline()
```

### 3. Walk-Forward Analysis

```python
# Run walk-forward analysis
wf_results = pipeline.run_walk_forward_analysis()

print(f"Average Return: {wf_results['avg_return']:.2%}")
print(f"Average Sharpe: {wf_results['avg_sharpe']:.3f}")
print(f"Positive Periods: {wf_results['positive_periods']}/{wf_results['n_periods']}")
```

## Output Files

The system generates comprehensive outputs in the specified directory:

### Data Files
- `sae_features.parquet`: Processed SAE features
- `trading_model.pkl`: Trained prediction model
- `config.json`: Configuration used

### Reports
- `backtest_report.txt`: Detailed performance report
- `results_summary.json`: Summary statistics
- `walk_forward_results.json`: Walk-forward analysis results

### Visualizations
- `plots/equity_curves.png`: Portfolio value over time
- `plots/drawdowns.png`: Drawdown comparison
- `plots/returns_distribution.png`: Returns distribution
- `plots/trade_analysis_*.png`: Individual strategy trade analysis

## Performance Metrics

### Risk-Adjusted Returns
- **Sharpe Ratio**: Risk-adjusted return measure
- **Sortino Ratio**: Downside risk-adjusted returns
- **Calmar Ratio**: Return vs maximum drawdown

### Risk Measures
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Value at Risk (VaR)**: Potential loss at 95% and 99% confidence
- **Conditional VaR (CVaR)**: Expected loss beyond VaR threshold

### Trading Statistics
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Average Trade Duration**: Mean holding period
- **Number of Trades**: Total trading activity

## Advanced Features

### Feature Engineering
- **Interaction Features**: Cross-products between SAE and technical features
- **Composite Features**: Layer-wise and type-wise aggregations
- **Feature Selection**: Automatic selection based on importance scores

### Model Ensemble
- **Voting**: Majority vote across models
- **Averaging**: Average prediction probabilities
- **Weighted Averaging**: Performance-weighted predictions

### Risk Management
- **Confidence Thresholding**: Only trade when model is confident
- **Position Sizing**: Dynamic position sizing based on confidence
- **Stop Loss**: Automatic exit conditions
- **Maximum Hold Period**: Prevent over-holding positions

## Troubleshooting

### Common Issues

1. **Data Fetching Errors**:
   - Check internet connection
   - Verify symbol names and timeframes
   - Check exchange rate limits

2. **Memory Issues**:
   - Reduce `days_back` parameter
   - Use higher timeframe (5m instead of 1m)
   - Reduce `n_features` parameter

3. **Model Training Errors**:
   - Ensure sufficient data (at least 1000 samples)
   - Check for NaN values in features
   - Verify target variable creation

### Performance Optimization

1. **Faster Execution**:
   - Use `5m` timeframe instead of `1m`
   - Reduce number of features
   - Use single model instead of ensemble

2. **Better Results**:
   - Increase `days_back` for more training data
   - Tune confidence thresholds
   - Experiment with different model types

## Contributing

To extend the system:

1. **Add New SAE Features**: Modify `sae_processor.py`
2. **Add Technical Indicators**: Extend `feature_engineer.py`
3. **Add New Models**: Extend `prediction_model.py`
4. **Add Risk Metrics**: Extend `backtesting_engine.py`

## License

This project is for educational and research purposes. Please ensure compliance with exchange terms of service and applicable regulations.

## Disclaimer

This trading system is for educational purposes only. Past performance does not guarantee future results. Trading cryptocurrencies involves substantial risk of loss. Always do your own research and consider consulting with a financial advisor before making investment decisions.

## References

- [VectorBT Documentation](https://vectorbt.dev/)
- [CCXT Documentation](https://docs.ccxt.com/)
- [AutoInterp Research](https://github.com/neelnanda-io/autointerp)
- [Sparse Autoencoders for Interpretability](https://arxiv.org/abs/2310.06674)
