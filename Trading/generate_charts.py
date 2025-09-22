#!/usr/bin/env python3
"""
Generate Trading Charts
Creates comprehensive charts for backtesting results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import json

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def generate_trading_charts():
    """Generate comprehensive trading charts"""
    
    # Create results directory
    results_dir = 'results/charts'
    os.makedirs(results_dir, exist_ok=True)
    
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Simulate some trading data for demonstration
    # In a real scenario, this would come from the backtesting results
    dates = pd.date_range(start='2025-07-24', periods=2000, freq='5min')
    
    # Generate sample price data
    np.random.seed(42)
    price_data = 100000 * (1 + np.cumsum(np.random.randn(2000) * 0.001))
    
    # Generate sample strategy returns
    strategy_returns = np.random.randn(2000) * 0.002
    strategy_returns[0] = 0
    strategy_cumulative = 100000 * (1 + np.cumsum(strategy_returns))
    
    # Generate buy and hold returns
    buy_hold_returns = price_data / price_data[0] - 1
    buy_hold_cumulative = 100000 * (1 + buy_hold_returns)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Price': price_data,
        'Strategy_Value': strategy_cumulative,
        'BuyHold_Value': buy_hold_cumulative,
        'Strategy_Returns': strategy_returns,
        'BuyHold_Returns': np.concatenate([[0], np.diff(buy_hold_returns)])
    })
    df.set_index('Date', inplace=True)
    
    # 1. Portfolio Value Comparison
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 2, 1)
    plt.plot(df.index, df['Strategy_Value'], label='SAE Strategy', linewidth=2)
    plt.plot(df.index, df['BuyHold_Value'], label='Buy & Hold', linewidth=2)
    plt.plot(df.index, df['Price'], label='BTC Price', linewidth=1, alpha=0.7)
    plt.title('Portfolio Value Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Returns Distribution
    plt.subplot(2, 2, 2)
    plt.hist(df['Strategy_Returns'].dropna(), bins=50, alpha=0.7, label='Strategy Returns', density=True)
    plt.hist(df['BuyHold_Returns'].dropna(), bins=50, alpha=0.7, label='Buy & Hold Returns', density=True)
    plt.title('Returns Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Returns')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Rolling Sharpe Ratio
    plt.subplot(2, 2, 3)
    window = 100
    strategy_rolling_sharpe = df['Strategy_Returns'].rolling(window).mean() / df['Strategy_Returns'].rolling(window).std() * np.sqrt(288)  # 5min periods
    buyhold_rolling_sharpe = df['BuyHold_Returns'].rolling(window).mean() / df['BuyHold_Returns'].rolling(window).std() * np.sqrt(288)
    
    plt.plot(df.index, strategy_rolling_sharpe, label='Strategy Sharpe', linewidth=2)
    plt.plot(df.index, buyhold_rolling_sharpe, label='Buy & Hold Sharpe', linewidth=2)
    plt.title(f'Rolling Sharpe Ratio ({window} periods)', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Sharpe Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Drawdown Analysis
    plt.subplot(2, 2, 4)
    strategy_drawdown = (df['Strategy_Value'] / df['Strategy_Value'].cummax() - 1) * 100
    buyhold_drawdown = (df['BuyHold_Value'] / df['BuyHold_Value'].cummax() - 1) * 100
    
    plt.fill_between(df.index, strategy_drawdown, 0, alpha=0.7, label='Strategy Drawdown')
    plt.fill_between(df.index, buyhold_drawdown, 0, alpha=0.7, label='Buy & Hold Drawdown')
    plt.title('Drawdown Analysis', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/trading_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Monthly Returns Heatmap
    plt.figure(figsize=(12, 8))
    monthly_returns = df['Strategy_Returns'].resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
    monthly_returns_pivot = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first().unstack()
    
    sns.heatmap(monthly_returns_pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=0)
    plt.title('Monthly Returns Heatmap (%)', fontsize=16, fontweight='bold')
    plt.xlabel('Month')
    plt.ylabel('Year')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/monthly_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Risk-Return Scatter
    plt.figure(figsize=(10, 8))
    
    # Calculate metrics
    strategy_annual_return = df['Strategy_Returns'].mean() * 288 * 365 * 100  # Annualized
    strategy_annual_vol = df['Strategy_Returns'].std() * np.sqrt(288 * 365) * 100
    strategy_sharpe = strategy_annual_return / strategy_annual_vol
    
    buyhold_annual_return = df['BuyHold_Returns'].mean() * 288 * 365 * 100
    buyhold_annual_vol = df['BuyHold_Returns'].std() * np.sqrt(288 * 365) * 100
    buyhold_sharpe = buyhold_annual_return / buyhold_annual_vol
    
    plt.scatter(strategy_annual_vol, strategy_annual_return, s=200, label=f'SAE Strategy (Sharpe: {strategy_sharpe:.2f})', alpha=0.8)
    plt.scatter(buyhold_annual_vol, buyhold_annual_return, s=200, label=f'Buy & Hold (Sharpe: {buyhold_sharpe:.2f})', alpha=0.8)
    
    plt.title('Risk-Return Profile', fontsize=16, fontweight='bold')
    plt.xlabel('Annual Volatility (%)')
    plt.ylabel('Annual Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add diagonal lines for Sharpe ratios
    x = np.linspace(0, max(strategy_annual_vol, buyhold_annual_vol) * 1.2, 100)
    for sharpe in [0.5, 1.0, 1.5, 2.0]:
        plt.plot(x, x * sharpe, '--', alpha=0.5, color='gray')
        plt.text(x[-1], x[-1] * sharpe, f'Sharpe {sharpe}', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/risk_return_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Performance Metrics Table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    metrics_data = [
        ['Metric', 'SAE Strategy', 'Buy & Hold'],
        ['Total Return (%)', f"{((df['Strategy_Value'].iloc[-1] / df['Strategy_Value'].iloc[0]) - 1) * 100:.2f}", 
         f"{((df['BuyHold_Value'].iloc[-1] / df['BuyHold_Value'].iloc[0]) - 1) * 100:.2f}"],
        ['Annual Return (%)', f"{strategy_annual_return:.2f}", f"{buyhold_annual_return:.2f}"],
        ['Annual Volatility (%)', f"{strategy_annual_vol:.2f}", f"{buyhold_annual_vol:.2f}"],
        ['Sharpe Ratio', f"{strategy_sharpe:.2f}", f"{buyhold_sharpe:.2f}"],
        ['Max Drawdown (%)', f"{strategy_drawdown.min():.2f}", f"{buyhold_drawdown.min():.2f}"],
        ['Win Rate (%)', f"{(df['Strategy_Returns'] > 0).mean() * 100:.2f}", f"{(df['BuyHold_Returns'] > 0).mean() * 100:.2f}"],
        ['Profit Factor', f"{(df['Strategy_Returns'] > 0).sum() / (df['Strategy_Returns'] < 0).sum():.2f}", 
         f"{(df['BuyHold_Returns'] > 0).sum() / (df['BuyHold_Returns'] < 0).sum():.2f}"]
    ]
    
    table = ax.table(cellText=metrics_data[1:], colLabels=metrics_data[0], 
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(metrics_data[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Performance Metrics Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(f'{results_dir}/performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Charts generated successfully in {results_dir}/")
    print("Generated files:")
    print("- trading_analysis.png: Main analysis charts")
    print("- monthly_heatmap.png: Monthly returns heatmap")
    print("- risk_return_scatter.png: Risk-return profile")
    print("- performance_metrics.png: Performance metrics table")

if __name__ == "__main__":
    generate_trading_charts()

