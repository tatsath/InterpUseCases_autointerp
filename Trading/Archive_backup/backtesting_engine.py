"""
Backtesting Engine using VectorBT
Comprehensive backtesting system for trading strategies with SAE features
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import List, Dict, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import quantstats as qs
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestingEngine:
    """Comprehensive backtesting engine using VectorBT"""
    
    def __init__(self, 
                 initial_cash: float = 100000,
                 fees: float = 0.0005,  # 5 bps
                 slippage: float = 0.0005,  # 5 bps
                 freq: str = '1min'):
        """
        Initialize the backtesting engine
        
        Args:
            initial_cash: Starting capital
            fees: Trading fees (as fraction)
            slippage: Slippage (as fraction)
            freq: Data frequency
        """
        self.initial_cash = initial_cash
        self.fees = fees
        self.slippage = slippage
        self.freq = freq
        
        # Results storage
        self.portfolios = {}
        self.results = {}
        self.metrics = {}
        
    def create_trading_signals(self, 
                             df: pd.DataFrame,
                             predictions: np.ndarray,
                             probabilities: Optional[np.ndarray] = None,
                             confidence_threshold: float = 0.6,
                             min_hold_period: int = 5,
                             max_hold_period: int = 60) -> pd.DataFrame:
        """
        Create trading signals from predictions
        
        Args:
            df: DataFrame with market data
            predictions: Model predictions (0/1)
            probabilities: Prediction probabilities
            confidence_threshold: Minimum confidence for trading
            min_hold_period: Minimum holding period
            max_hold_period: Maximum holding period
            
        Returns:
            DataFrame with trading signals
        """
        signals_df = df.copy()
        
        # Ensure predictions and probabilities match the dataframe length
        if len(predictions) != len(signals_df):
            # If predictions are shorter, pad with zeros (no signal)
            padded_predictions = np.zeros(len(signals_df))
            padded_probabilities = np.zeros(len(signals_df))
            padded_predictions[:len(predictions)] = predictions
            padded_probabilities[:len(predictions)] = probabilities if probabilities is not None else np.ones_like(predictions) * 0.5
            signals_df['predictions'] = padded_predictions
            signals_df['probabilities'] = padded_probabilities
        else:
            signals_df['predictions'] = predictions
            signals_df['probabilities'] = probabilities if probabilities is not None else np.ones_like(predictions) * 0.5
        
        # Create entry signals (only when confident)
        signals_df['entry_signal'] = (
            (signals_df['predictions'] == 1) & 
            (signals_df['probabilities'] >= confidence_threshold)
        ).astype(int)
        
        # Create exit signals based on various conditions
        signals_df['exit_signal'] = 0
        
        # Exit after max hold period
        signals_df['hold_count'] = 0
        in_position = False
        hold_count = 0
        
        for i in range(len(signals_df)):
            if signals_df['entry_signal'].iloc[i] and not in_position:
                in_position = True
                hold_count = 0
            elif in_position:
                hold_count += 1
                
                # Exit conditions
                should_exit = (
                    hold_count >= max_hold_period or  # Max hold time
                    (hold_count >= min_hold_period and signals_df['predictions'].iloc[i] == 0) or  # Signal reversal
                    signals_df['probabilities'].iloc[i] < confidence_threshold  # Low confidence
                )
                
                if should_exit:
                    signals_df['exit_signal'].iloc[i] = 1
                    in_position = False
                    hold_count = 0
                else:
                    signals_df['hold_count'].iloc[i] = hold_count
        
        # Create position signals
        signals_df['position'] = 0
        position = 0
        
        for i in range(len(signals_df)):
            if signals_df['entry_signal'].iloc[i]:
                position = 1
            elif signals_df['exit_signal'].iloc[i]:
                position = 0
            
            signals_df['position'].iloc[i] = position
        
        logger.info(f"Created signals: {signals_df['entry_signal'].sum()} entries, {signals_df['exit_signal'].sum()} exits")
        
        return signals_df
    
    def run_backtest(self, 
                    df: pd.DataFrame,
                    signals_df: pd.DataFrame,
                    strategy_name: str = 'strategy',
                    symbol: str = 'BTC/USDT') -> Dict[str, Any]:
        """
        Run backtest for a single strategy
        
        Args:
            df: DataFrame with OHLCV data
            signals_df: DataFrame with trading signals
            strategy_name: Name of the strategy
            symbol: Trading symbol
            
        Returns:
            Dictionary with backtest results
        """
        try:
            # Prepare data - use close_x if available, otherwise close
            price_col = 'close_x' if 'close_x' in df.columns else 'close'
            price = df[price_col]
            entries = signals_df['entry_signal'].astype(bool)
            exits = signals_df['exit_signal'].astype(bool)
            
            # Create portfolio
            portfolio = vbt.Portfolio.from_signals(
                price,
                entries,
                exits,
                freq=self.freq,
                fees=self.fees,
                slippage=self.slippage,
                init_cash=self.initial_cash,
                size=np.inf,  # Use all available cash
                size_type='value'
            )
            
            # Store portfolio
            self.portfolios[strategy_name] = portfolio
            
            # Calculate metrics
            metrics = self._calculate_metrics(portfolio, price, entries, exits)
            self.metrics[strategy_name] = metrics
            
            # Store results
            results = {
                'portfolio': portfolio,
                'metrics': metrics,
                'signals': signals_df,
                'price_data': df
            }
            
            self.results[strategy_name] = results
            
            logger.info(f"Backtest completed for {strategy_name}")
            logger.info(f"Total Return: {metrics['total_return']:.2%}")
            logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in backtest for {strategy_name}: {str(e)}")
            return {}
    
    def run_comparison_backtest(self, 
                              df: pd.DataFrame,
                              signals_df: pd.DataFrame,
                              baseline_signals: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Run backtest comparing multiple strategies
        
        Args:
            df: DataFrame with OHLCV data
            signals_df: DataFrame with SAE-based signals
            baseline_signals: DataFrame with baseline signals
            
        Returns:
            Dictionary with comparison results
        """
        comparison_results = {}
        
        # SAE-based strategy
        sae_results = self.run_backtest(df, signals_df, 'sae_strategy')
        comparison_results['sae_strategy'] = sae_results
        
        # Baseline strategy (if provided)
        if baseline_signals is not None:
            baseline_results = self.run_backtest(df, baseline_signals, 'baseline_strategy')
            comparison_results['baseline_strategy'] = baseline_results
        
        # Buy and hold strategy
        buy_hold_signals = self._create_buy_hold_signals(df)
        buy_hold_results = self.run_backtest(df, buy_hold_signals, 'buy_hold')
        comparison_results['buy_hold'] = buy_hold_results
        
        return comparison_results
    
    def _create_buy_hold_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create buy and hold signals for comparison"""
        signals_df = pd.DataFrame(index=df.index)
        signals_df['entry_signal'] = 0
        signals_df['exit_signal'] = 0
        signals_df['predictions'] = 1
        signals_df['probabilities'] = 1.0
        signals_df['position'] = 1
        
        # Buy at the beginning, sell at the end
        signals_df['entry_signal'].iloc[0] = 1
        signals_df['exit_signal'].iloc[-1] = 1
        
        return signals_df
    
    def _calculate_metrics(self, 
                          portfolio: vbt.Portfolio, 
                          price: pd.Series,
                          entries: pd.Series,
                          exits: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        # Basic metrics
        total_return = portfolio.total_return()
        annualized_return = portfolio.annualized_return()
        sharpe_ratio = portfolio.sharpe_ratio()
        max_drawdown = portfolio.max_drawdown()
        
        # Additional metrics
        win_rate = portfolio.trades.win_rate()
        profit_factor = portfolio.trades.profit_factor()
        avg_trade_duration = portfolio.trades.duration.mean()
        
        # Risk metrics
        volatility = portfolio.annualized_volatility()
        sortino_ratio = portfolio.sortino_ratio()
        calmar_ratio = portfolio.calmar_ratio()
        
        # Trade statistics
        n_trades = len(portfolio.trades.records_readable)
        avg_trade_pnl = portfolio.trades.pnl.mean() if n_trades > 0 else 0
        max_trade_pnl = portfolio.trades.pnl.max() if n_trades > 0 else 0
        min_trade_pnl = portfolio.trades.pnl.min() if n_trades > 0 else 0
        
        # Value at Risk (VaR)
        returns = portfolio.returns()
        var_95 = np.percentile(returns.dropna(), 5)
        var_99 = np.percentile(returns.dropna(), 1)
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = returns[returns <= var_95].mean() if not pd.isna(var_95) else 0
        cvar_99 = returns[returns <= var_99].mean() if not pd.isna(var_99) else 0
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade_duration': avg_trade_duration,
            'n_trades': n_trades,
            'avg_trade_pnl': avg_trade_pnl,
            'max_trade_pnl': max_trade_pnl,
            'min_trade_pnl': min_trade_pnl,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99
        }
        
        return metrics
    
    def generate_report(self, 
                       results: Dict[str, Any],
                       save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive backtest report
        
        Args:
            results: Backtest results
            save_path: Path to save report
            
        Returns:
            Report text
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("BACKTESTING REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Strategy comparison
        if len(results) > 1:
            report_lines.append("STRATEGY COMPARISON")
            report_lines.append("-" * 40)
            
            comparison_data = []
            for strategy_name, strategy_results in results.items():
                if 'metrics' in strategy_results:
                    metrics = strategy_results['metrics']
                    comparison_data.append({
                        'Strategy': strategy_name,
                        'Total Return': f"{metrics['total_return']:.2%}",
                        'Sharpe Ratio': f"{metrics['sharpe_ratio']:.3f}",
                        'Max Drawdown': f"{metrics['max_drawdown']:.2%}",
                        'Win Rate': f"{metrics['win_rate']:.2%}",
                        'N Trades': f"{metrics['n_trades']:.0f}"
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            report_lines.append(comparison_df.to_string(index=False))
            report_lines.append("")
        
        # Detailed metrics for each strategy
        for strategy_name, strategy_results in results.items():
            if 'metrics' in strategy_results:
                report_lines.append(f"DETAILED METRICS - {strategy_name.upper()}")
                report_lines.append("-" * 40)
                
                metrics = strategy_results['metrics']
                
                # Performance metrics
                report_lines.append("Performance Metrics:")
                report_lines.append(f"  Total Return: {metrics['total_return']:.2%}")
                report_lines.append(f"  Annualized Return: {metrics['annualized_return']:.2%}")
                report_lines.append(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
                report_lines.append(f"  Sortino Ratio: {metrics['sortino_ratio']:.3f}")
                report_lines.append(f"  Calmar Ratio: {metrics['calmar_ratio']:.3f}")
                report_lines.append("")
                
                # Risk metrics
                report_lines.append("Risk Metrics:")
                report_lines.append(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
                report_lines.append(f"  Volatility: {metrics['volatility']:.2%}")
                report_lines.append(f"  VaR (95%): {metrics['var_95']:.2%}")
                report_lines.append(f"  VaR (99%): {metrics['var_99']:.2%}")
                report_lines.append(f"  CVaR (95%): {metrics['cvar_95']:.2%}")
                report_lines.append(f"  CVaR (99%): {metrics['cvar_99']:.2%}")
                report_lines.append("")
                
                # Trade metrics
                report_lines.append("Trade Metrics:")
                report_lines.append(f"  Number of Trades: {metrics['n_trades']:.0f}")
                report_lines.append(f"  Win Rate: {metrics['win_rate']:.2%}")
                report_lines.append(f"  Profit Factor: {metrics['profit_factor']:.3f}")
                report_lines.append(f"  Avg Trade Duration: {metrics['avg_trade_duration']:.1f} periods")
                report_lines.append(f"  Avg Trade P&L: {metrics['avg_trade_pnl']:.2f}")
                report_lines.append(f"  Max Trade P&L: {metrics['max_trade_pnl']:.2f}")
                report_lines.append(f"  Min Trade P&L: {metrics['min_trade_pnl']:.2f}")
                report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {save_path}")
        
        return report_text
    
    def create_plots(self, 
                    results: Dict[str, Any],
                    save_dir: str = 'plots') -> List[str]:
        """
        Create visualization plots
        
        Args:
            results: Backtest results
            save_dir: Directory to save plots
            
        Returns:
            List of saved plot paths
        """
        os.makedirs(save_dir, exist_ok=True)
        plot_paths = []
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Equity curve comparison
        if len(results) > 1:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for strategy_name, strategy_results in results.items():
                if 'portfolio' in strategy_results:
                    portfolio = strategy_results['portfolio']
                    equity = portfolio.value()
                    ax.plot(equity.index, equity.values, label=strategy_name, linewidth=2)
            
            ax.set_title('Equity Curve Comparison', fontsize=16, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Portfolio Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_path = os.path.join(save_dir, 'equity_curves.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plot_paths.append(plot_path)
            plt.close()
        
        # 2. Drawdown comparison
        if len(results) > 1:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for strategy_name, strategy_results in results.items():
                if 'portfolio' in strategy_results:
                    portfolio = strategy_results['portfolio']
                    drawdown = portfolio.drawdowns.drawdown
                    # Convert to pandas Series if it's a MappedArray
                    if hasattr(drawdown, 'index'):
                        drawdown_index = drawdown.index
                        drawdown_values = drawdown.values
                    else:
                        # Use the portfolio's value index
                        drawdown_index = portfolio.value().index
                        drawdown_values = drawdown
                    ax.fill_between(drawdown_index, drawdown_values, 0, 
                                  alpha=0.7, label=strategy_name)
            
            ax.set_title('Drawdown Comparison', fontsize=16, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Drawdown')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_path = os.path.join(save_dir, 'drawdowns.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plot_paths.append(plot_path)
            plt.close()
        
        # 3. Returns distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (strategy_name, strategy_results) in enumerate(results.items()):
            if i >= 4:  # Limit to 4 strategies
                break
                
            if 'portfolio' in strategy_results:
                portfolio = strategy_results['portfolio']
                returns = portfolio.returns().dropna()
                
                # Histogram
                axes[i].hist(returns, bins=50, alpha=0.7, density=True)
                axes[i].set_title(f'{strategy_name} - Returns Distribution')
                axes[i].set_xlabel('Returns')
                axes[i].set_ylabel('Density')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(results), 4):
            axes[i].set_visible(False)
        
        plot_path = os.path.join(save_dir, 'returns_distribution.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plot_paths.append(plot_path)
        plt.close()
        
        # 4. Trade analysis
        for strategy_name, strategy_results in results.items():
            if 'portfolio' in strategy_results:
                portfolio = strategy_results['portfolio']
                
                if len(portfolio.trades.records_readable) > 0:
                    trades = portfolio.trades.records_readable
                    
                    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                    
                    # P&L distribution
                    axes[0, 0].hist(trades['PnL'], bins=30, alpha=0.7)
                    axes[0, 0].set_title(f'{strategy_name} - Trade P&L Distribution')
                    axes[0, 0].set_xlabel('P&L')
                    axes[0, 0].set_ylabel('Frequency')
                    axes[0, 0].grid(True, alpha=0.3)
                    
                    # Trade duration
                    axes[0, 1].hist(trades['Duration'], bins=30, alpha=0.7)
                    axes[0, 1].set_title(f'{strategy_name} - Trade Duration')
                    axes[0, 1].set_xlabel('Duration (periods)')
                    axes[0, 1].set_ylabel('Frequency')
                    axes[0, 1].grid(True, alpha=0.3)
                    
                    # P&L over time
                    axes[1, 0].plot(trades['Entry Timestamp'], trades['PnL'], 'o-', alpha=0.7)
                    axes[1, 0].set_title(f'{strategy_name} - P&L Over Time')
                    axes[1, 0].set_xlabel('Entry Time')
                    axes[1, 0].set_ylabel('P&L')
                    axes[1, 0].grid(True, alpha=0.3)
                    
                    # Win/Loss ratio
                    win_loss = trades['PnL'] > 0
                    win_loss_counts = win_loss.value_counts()
                    axes[1, 1].pie(win_loss_counts.values, labels=['Loss', 'Win'], autopct='%1.1f%%')
                    axes[1, 1].set_title(f'{strategy_name} - Win/Loss Ratio')
                    
                    plt.tight_layout()
                    plot_path = os.path.join(save_dir, f'trade_analysis_{strategy_name}.png')
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plot_paths.append(plot_path)
                    plt.close()
        
        logger.info(f"Created {len(plot_paths)} plots in {save_dir}")
        return plot_paths
    
    def run_walk_forward_analysis(self, 
                                df: pd.DataFrame,
                                signals_df: pd.DataFrame,
                                train_period: int = 252,  # 1 year
                                test_period: int = 63,    # 3 months
                                step_size: int = 21) -> Dict[str, Any]:
        """
        Run walk-forward analysis
        
        Args:
            df: DataFrame with market data
            signals_df: DataFrame with signals
            train_period: Training period length
            test_period: Test period length
            step_size: Step size for rolling window
            
        Returns:
            Dictionary with walk-forward results
        """
        walk_forward_results = []
        
        # Create rolling windows
        start_idx = 0
        while start_idx + train_period + test_period < len(df):
            train_end = start_idx + train_period
            test_end = train_end + test_period
            
            # Split data
            train_data = df.iloc[start_idx:train_end]
            test_data = df.iloc[train_end:test_end]
            train_signals = signals_df.iloc[start_idx:train_end]
            test_signals = signals_df.iloc[train_end:test_end]
            
            # Run backtest on test period
            test_results = self.run_backtest(
                test_data, 
                test_signals, 
                f'walk_forward_{start_idx}'
            )
            
            if test_results:
                walk_forward_results.append({
                    'period': f'{start_idx}_{test_end}',
                    'train_start': train_data.index[0],
                    'train_end': train_data.index[-1],
                    'test_start': test_data.index[0],
                    'test_end': test_data.index[-1],
                    'metrics': test_results['metrics']
                })
            
            start_idx += step_size
        
        # Aggregate results
        if walk_forward_results:
            all_returns = []
            all_sharpe = []
            all_drawdowns = []
            
            for result in walk_forward_results:
                metrics = result['metrics']
                all_returns.append(metrics['total_return'])
                all_sharpe.append(metrics['sharpe_ratio'])
                all_drawdowns.append(metrics['max_drawdown'])
            
            walk_forward_summary = {
                'n_periods': len(walk_forward_results),
                'avg_return': np.mean(all_returns),
                'std_return': np.std(all_returns),
                'avg_sharpe': np.mean(all_sharpe),
                'std_sharpe': np.std(all_sharpe),
                'avg_drawdown': np.mean(all_drawdowns),
                'max_drawdown': np.max(all_drawdowns),
                'positive_periods': sum(1 for r in all_returns if r > 0),
                'detailed_results': walk_forward_results
            }
            
            logger.info(f"Walk-forward analysis completed: {len(walk_forward_results)} periods")
            logger.info(f"Average Return: {walk_forward_summary['avg_return']:.2%}")
            logger.info(f"Average Sharpe: {walk_forward_summary['avg_sharpe']:.3f}")
            
            return walk_forward_summary
        
        return {}

def main():
    """Example usage of the backtesting engine"""
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    
    # Create synthetic price data
    price = 50000 + np.cumsum(np.random.randn(1000) * 10)
    
    sample_data = pd.DataFrame({
        'open': price + np.random.randn(1000) * 5,
        'high': price + np.abs(np.random.randn(1000) * 10),
        'low': price - np.abs(np.random.randn(1000) * 10),
        'close': price,
        'volume': np.random.exponential(1000, 1000)
    }, index=dates)
    
    # Create sample signals
    signals = np.random.choice([0, 1], size=1000, p=[0.8, 0.2])
    probabilities = np.random.uniform(0.5, 1.0, 1000)
    
    signals_df = pd.DataFrame({
        'entry_signal': signals,
        'exit_signal': np.roll(signals, 10),  # Exit 10 periods later
        'predictions': signals,
        'probabilities': probabilities,
        'position': signals
    }, index=dates)
    
    # Initialize backtesting engine
    engine = BacktestingEngine()
    
    # Run backtest
    results = engine.run_backtest(sample_data, signals_df, 'sample_strategy')
    
    if results:
        print("Backtest completed successfully!")
        print(f"Total Return: {results['metrics']['total_return']:.2%}")
        print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.3f}")
        
        # Generate report
        report = engine.generate_report({'sample_strategy': results})
        print("\n" + report)

if __name__ == "__main__":
    main()
