"""
Main Trading Script
Orchestrates the complete trading pipeline with SAE features
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import argparse
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

class TradingPipeline:
    """Main trading pipeline that orchestrates all components"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize the trading pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.data_fetcher = CryptoDataFetcher()
        self.sae_processor = SAEFeatureProcessor()
        self.feature_engineer = FeatureEngineer()
        self.prediction_model = TradingPredictionModel(
            model_type=self.config['model_type']
        )
        self.backtesting_engine = BacktestingEngine(
            initial_cash=self.config['initial_cash'],
            fees=self.config['fees'],
            slippage=self.config['slippage']
        )
        
        # Results storage
        self.results = {}
        self.data = {}
        
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'symbols': ['BTC/USDT', 'DOGE/USDT'],
            'timeframe': '1m',
            'days_back': 30,
            'model_type': 'ensemble',
            'initial_cash': 100000,
            'fees': 0.0005,
            'slippage': 0.0005,
            'confidence_threshold': 0.6,
            'min_hold_period': 5,
            'max_hold_period': 60,
            'test_size': 0.2,
            'validation_size': 0.1,
            'n_features': 50,
            'save_results': True,
            'output_dir': 'results'
        }
    
    def run_complete_pipeline(self) -> Dict[str, any]:
        """
        Run the complete trading pipeline
        
        Returns:
            Dictionary with all results
        """
        logger.info("Starting complete trading pipeline...")
        
        try:
            # Step 1: Fetch market data
            logger.info("Step 1: Fetching market data...")
            market_data = self._fetch_market_data()
            self.data['market'] = market_data
            
            # Step 2: Process SAE features
            logger.info("Step 2: Processing SAE features...")
            sae_data = self._process_sae_features(market_data)
            self.data['sae'] = sae_data
            
            # Step 3: Engineer features
            logger.info("Step 3: Engineering features...")
            engineered_data = self._engineer_features(market_data, sae_data)
            self.data['engineered'] = engineered_data
            
            # Step 4: Train prediction model
            logger.info("Step 4: Training prediction model...")
            model_results = self._train_prediction_model(engineered_data)
            self.results['model'] = model_results
            
            # Step 5: Create trading signals
            logger.info("Step 5: Creating trading signals...")
            signals = self._create_trading_signals(engineered_data, model_results)
            self.data['signals'] = signals
            
            # Step 6: Run backtesting
            logger.info("Step 6: Running backtesting...")
            backtest_results = self._run_backtesting(market_data, signals)
            self.results['backtest'] = backtest_results
            
            # Step 7: Generate reports
            logger.info("Step 7: Generating reports...")
            self._generate_reports(backtest_results)
            
            logger.info("Trading pipeline completed successfully!")
            return self.results
            
        except Exception as e:
            logger.error(f"Error in trading pipeline: {str(e)}")
            raise
    
    def _fetch_market_data(self) -> pd.DataFrame:
        """Fetch market data for all symbols"""
        all_data = []
        
        for symbol in self.config['symbols']:
            logger.info(f"Fetching data for {symbol}...")
            
            # Fetch data
            df = self.data_fetcher.fetch_ohlcv(
                symbol=symbol,
                timeframe=self.config['timeframe'],
                since=datetime.now() - timedelta(days=self.config['days_back'])
            )
            
            if not df.empty:
                df['symbol'] = symbol.replace('/', '')
                all_data.append(df)
            else:
                logger.warning(f"No data fetched for {symbol}")
        
        if not all_data:
            raise ValueError("No market data fetched for any symbol")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=False)
        combined_data = combined_data.sort_index()
        
        logger.info(f"Fetched {len(combined_data)} candles for {len(self.config['symbols'])} symbols")
        return combined_data
    
    def _process_sae_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Process SAE features for the market data"""
        # Create SAE features
        sae_data = self.sae_processor.create_sae_features(market_data)
        
        # Save SAE features
        if self.config['save_results']:
            os.makedirs(self.config['output_dir'], exist_ok=True)
            sae_path = os.path.join(self.config['output_dir'], 'sae_features.parquet')
            self.sae_processor.save_sae_features(sae_data, 'sae_features.parquet', self.config['output_dir'])
        
        return sae_data
    
    def _engineer_features(self, market_data: pd.DataFrame, sae_data: pd.DataFrame) -> pd.DataFrame:
        """Engineer all features"""
        # Create baseline technical features
        baseline_data = self.feature_engineer.create_baseline_features(market_data)
        
        # Integrate SAE features
        combined_data = self.feature_engineer.integrate_sae_features(baseline_data, sae_data)
        
        # Create target variable (use close_x as the price column after merge)
        price_col = 'close_x' if 'close_x' in combined_data.columns else 'close'
        combined_data = self.prediction_model.create_target_variable(combined_data, price_col=price_col)
        
        return combined_data
    
    def _train_prediction_model(self, data: pd.DataFrame) -> Dict[str, any]:
        """Train the prediction model"""
        # Get feature columns (exclude all non-numeric columns)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'symbol', 
                       'returns', 'log_returns', 'target', 'target_3class', 'target_continuous', 'target_vol_adj',
                       'open_x', 'high_x', 'low_x', 'close_x', 'volume_x', 'symbol_x', 'returns_x', 'log_returns_x',
                       'open_y', 'high_y', 'low_y', 'close_y', 'volume_y', 'symbol_y', 'returns_y', 'log_returns_y']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        # Additional filter to ensure only numeric columns are included
        numeric_cols = data[feature_cols].select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in feature_cols if col in numeric_cols]
        
        # Prepare training data
        data_splits = self.prediction_model.prepare_training_data(
            data, 
            feature_cols,
            test_size=self.config['test_size'],
            validation_size=self.config['validation_size']
        )
        
        # Train models
        training_results = self.prediction_model.train_models(data_splits)
        
        # Evaluate models
        evaluation_results = self.prediction_model.evaluate_models(data_splits)
        
        # Get feature importance
        feature_importance = self.prediction_model.get_feature_importance()
        
        model_results = {
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'feature_importance': feature_importance,
            'data_splits': data_splits
        }
        
        # Save model
        if self.config['save_results']:
            model_path = os.path.join(self.config['output_dir'], 'trading_model.pkl')
            self.prediction_model.save_model(model_path)
        
        return model_results
    
    def _create_trading_signals(self, data: pd.DataFrame, model_results: Dict) -> pd.DataFrame:
        """Create trading signals using the trained model"""
        # Get test data
        test_data = model_results['data_splits']['X_test']
        
        # Make predictions
        predictions = self.prediction_model.predict(test_data, model_name='ensemble')
        
        # Create signals DataFrame
        signals_df = pd.DataFrame(index=test_data.index)
        signals_df['predictions'] = predictions['predictions']
        signals_df['probabilities'] = predictions['probabilities'] if predictions['probabilities'] is not None else np.ones_like(predictions['predictions']) * 0.5
        
        # Create trading signals
        signals_df = self.backtesting_engine.create_trading_signals(
            data.loc[test_data.index],
            signals_df['predictions'].values,
            signals_df['probabilities'].values,
            confidence_threshold=self.config['confidence_threshold'],
            min_hold_period=self.config['min_hold_period'],
            max_hold_period=self.config['max_hold_period']
        )
        
        return signals_df
    
    def _run_backtesting(self, market_data: pd.DataFrame, signals: pd.DataFrame) -> Dict[str, any]:
        """Run comprehensive backtesting"""
        # Get test data for backtesting
        test_start = signals.index[0]
        test_end = signals.index[-1]
        test_market_data = market_data.loc[test_start:test_end]
        
        # Create baseline signals for comparison
        baseline_signals = self._create_baseline_signals(test_market_data)
        
        # Run comparison backtest
        backtest_results = self.backtesting_engine.run_comparison_backtest(
            test_market_data,
            signals,
            baseline_signals
        )
        
        return backtest_results
    
    def _create_baseline_signals(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Create baseline trading signals for comparison"""
        # Simple EMA crossover strategy
        ema_fast = market_data['close'].ewm(span=20).mean()
        ema_slow = market_data['close'].ewm(span=50).mean()
        
        # Create signals
        signals_df = pd.DataFrame(index=market_data.index)
        signals_df['predictions'] = (ema_fast > ema_slow).astype(int)
        signals_df['probabilities'] = np.where(signals_df['predictions'] == 1, 0.7, 0.3)
        
        # Create trading signals
        signals_df = self.backtesting_engine.create_trading_signals(
            market_data,
            signals_df['predictions'].values,
            signals_df['probabilities'].values,
            confidence_threshold=0.5,
            min_hold_period=10,
            max_hold_period=120
        )
        
        return signals_df
    
    def _generate_reports(self, backtest_results: Dict[str, any]):
        """Generate comprehensive reports"""
        if not self.config['save_results']:
            return
        
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Generate text report
        report_text = self.backtesting_engine.generate_report(backtest_results)
        report_path = os.path.join(self.config['output_dir'], 'backtest_report.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        # Generate plots
        plots_dir = os.path.join(self.config['output_dir'], 'plots')
        plot_paths = self.backtesting_engine.create_plots(backtest_results, plots_dir)
        
        # Save configuration
        config_path = os.path.join(self.config['output_dir'], 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
        
        # Save results summary
        summary = self._create_results_summary(backtest_results)
        summary_path = os.path.join(self.config['output_dir'], 'results_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Reports generated in {self.config['output_dir']}")
        logger.info(f"Plots saved: {len(plot_paths)} files")
    
    def _create_results_summary(self, backtest_results: Dict[str, any]) -> Dict[str, any]:
        """Create a summary of results"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'strategies': {}
        }
        
        for strategy_name, strategy_results in backtest_results.items():
            if 'metrics' in strategy_results:
                metrics = strategy_results['metrics']
                summary['strategies'][strategy_name] = {
                    'total_return': float(metrics['total_return']),
                    'sharpe_ratio': float(metrics['sharpe_ratio']),
                    'max_drawdown': float(metrics['max_drawdown']),
                    'win_rate': float(metrics['win_rate']),
                    'n_trades': int(metrics['n_trades']),
                    'volatility': float(metrics['volatility'])
                }
        
        return summary
    
    def run_walk_forward_analysis(self) -> Dict[str, any]:
        """Run walk-forward analysis"""
        logger.info("Running walk-forward analysis...")
        
        # Get all data
        market_data = self.data.get('market')
        if market_data is None:
            market_data = self._fetch_market_data()
        
        sae_data = self.data.get('sae')
        if sae_data is None:
            sae_data = self._process_sae_features(market_data)
        
        engineered_data = self.data.get('engineered')
        if engineered_data is None:
            engineered_data = self._engineer_features(market_data, sae_data)
        
        # Run walk-forward analysis
        walk_forward_results = self.backtesting_engine.run_walk_forward_analysis(
            engineered_data,
            engineered_data,  # Using engineered data as signals for simplicity
            train_period=252,  # 1 year
            test_period=63,    # 3 months
            step_size=21       # 1 month
        )
        
        # Save walk-forward results
        if self.config['save_results'] and walk_forward_results:
            wf_path = os.path.join(self.config['output_dir'], 'walk_forward_results.json')
            with open(wf_path, 'w') as f:
                json.dump(walk_forward_results, f, indent=2, default=str)
        
        return walk_forward_results

def main():
    """Main function to run the trading pipeline"""
    parser = argparse.ArgumentParser(description='Run SAE-based trading pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--symbols', nargs='+', default=['BTC/USDT', 'DOGE/USDT'], help='Trading symbols')
    parser.add_argument('--timeframe', type=str, default='1m', help='Data timeframe')
    parser.add_argument('--days', type=int, default=30, help='Days of historical data')
    parser.add_argument('--model', type=str, default='ensemble', help='Model type')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--walk-forward', action='store_true', help='Run walk-forward analysis')
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Override with command line arguments
    if config is None:
        config = {}
    
    config.update({
        'symbols': args.symbols,
        'timeframe': args.timeframe,
        'days_back': args.days,
        'model_type': args.model,
        'output_dir': args.output
    })
    
    # Initialize and run pipeline
    pipeline = TradingPipeline(config)
    
    try:
        # Run main pipeline
        results = pipeline.run_complete_pipeline()
        
        # Run walk-forward analysis if requested
        if args.walk_forward:
            wf_results = pipeline.run_walk_forward_analysis()
            results['walk_forward'] = wf_results
        
        print("\n" + "="*80)
        print("TRADING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        # Print summary
        if 'backtest' in results:
            for strategy_name, strategy_results in results['backtest'].items():
                if 'metrics' in strategy_results:
                    metrics = strategy_results['metrics']
                    print(f"\n{strategy_name.upper()}:")
                    print(f"  Total Return: {metrics['total_return']:.2%}")
                    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
                    print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
                    print(f"  Win Rate: {metrics['win_rate']:.2%}")
                    print(f"  Number of Trades: {metrics['n_trades']:.0f}")
        
        print(f"\nResults saved to: {config['output_dir']}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()


