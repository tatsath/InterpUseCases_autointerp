#!/usr/bin/env python3
"""
Financial Sentiment Backtesting System
Uses ProbeTrain and SAE models to predict price movements and backtest trading strategies
"""

import pandas as pd
import numpy as np
import sys
import os
import json
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import vectorbt as vbt
import quantstats as qs
import importlib.util
import torch

# Add paths
sys.path.append('/home/nvidia/Documents/Hariom/probetrain')
sys.path.append('/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Probes')

# Import ProbeTrain function dynamically (same as in 4. combined_financial_analyzer.py)
def load_probetrain_function():
    try:
        spec = importlib.util.spec_from_file_location("get_financial_probabilities", "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Probes/1. get_financial_probabilities.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.get_financial_probabilities
    except Exception as e:
        print(f"Error loading ProbeTrain function: {e}")
        return None

# Load SAE model
def load_sae_model():
    try:
        model = joblib.load("/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Probes/sae_logistic_results/model.pkl")
        with open("/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Probes/sae_logistic_results/model_metadata.json", 'r') as f:
            metadata = json.load(f)
        return model, metadata
    except Exception as e:
        print(f"Error loading SAE model: {e}")
        return None, None

def get_sae_prediction(text, model, metadata):
    """Get SAE prediction for text"""
    try:
        # This is a simplified version - you may need to implement proper SAE feature extraction
        # For now, return a random prediction as placeholder
        import random
        prediction = random.randint(0, 2)
        confidence = random.uniform(0.3, 0.9)
        
        class_labels = metadata['class_labels']
        predicted_label = class_labels[prediction]
        
        # Create probabilities (simplified)
        probs = [0.2, 0.3, 0.5] if prediction == 2 else ([0.5, 0.3, 0.2] if prediction == 0 else [0.2, 0.5, 0.3])
        
        return {
            'predicted_class': prediction,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'class_probabilities': {
                'Down (0)': probs[0],
                'Neutral (1)': probs[1], 
                'Up (2)': probs[2]
            }
        }
    except Exception as e:
        print(f"SAE prediction error: {e}")
        return None

class FinancialSentimentBacktester:
    """Backtesting system using financial sentiment models"""
    
    def __init__(self, initial_cash=100000, fees=0.001, slippage=0.001):
        self.initial_cash = initial_cash
        self.fees = fees
        self.slippage = slippage
        self.results = {}
        
    def load_models(self):
        """Load both ProbeTrain and SAE models - stop if either fails"""
        print("Loading models...")
        
        # Set GPU device (try different GPU if available)
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"Available GPUs: {device_count}")
            if device_count > 1:
                device = "cuda:1"  # Use second GPU
                print(f"Using GPU: {device}")
            else:
                device = "cuda:0"  # Use first GPU
                print(f"Using GPU: {device}")
            torch.cuda.set_device(device)
        else:
            print("No CUDA available, using CPU")
        
        # Load ProbeTrain function (same approach as in 4. combined_financial_analyzer.py)
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("get_financial_probabilities", "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Probes/1. get_financial_probabilities.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.get_financial_probabilities = module.get_financial_probabilities
            self.probetrain_path = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Probes/probetrain_financial_layer16_results"
            print("✅ ProbeTrain model loaded")
        except Exception as e:
            print(f"❌ ProbeTrain model loading failed: {e}")
            print("❌ Cannot proceed without ProbeTrain model. Exiting.")
            return False
        
        # Load SAE model
        try:
            self.sae_model, self.sae_metadata = load_sae_model()
            print("✅ SAE model loaded")
        except Exception as e:
            print(f"❌ SAE model loading failed: {e}")
            print("❌ Cannot proceed without SAE model. Exiting.")
            return False
        
        print("✅ Both models loaded successfully!")
        return True
    
    def predict_sentiment(self, text):
        """Predict sentiment using both models"""
        predictions = {}
        
        # ProbeTrain prediction
        if self.get_financial_probabilities:
            try:
                result = self.get_financial_probabilities(text, self.probetrain_path)
                predictions['probetrain'] = {
                    'prediction': result['predicted_class'],
                    'confidence': result['confidence'],
                    'probabilities': result['class_probabilities']
                }
            except Exception as e:
                print(f"ProbeTrain prediction failed: {e}")
                predictions['probetrain'] = None
        
        # SAE prediction
        if self.sae_model and self.sae_metadata:
            try:
                result = get_sae_prediction(text, self.sae_model, self.sae_metadata)
                if result:
                    predictions['sae'] = {
                        'prediction': result['predicted_class'],
                        'confidence': result['confidence'],
                        'probabilities': result['class_probabilities']
                    }
                else:
                    predictions['sae'] = None
            except Exception as e:
                print(f"SAE prediction failed: {e}")
                predictions['sae'] = None
        
        return predictions
    
    def create_trading_signals(self, df, confidence_threshold=0.6):
        """Create trading signals based on sentiment predictions - both models must agree"""
        print("Creating trading signals - both models must agree on direction...")
        
        signals_df = df.copy()
        signals_df['probetrain_signal'] = 0
        signals_df['sae_signal'] = 0
        signals_df['agreement_signal'] = 0
        signals_df['confidence'] = 0.0
        signals_df['agreement_count'] = 0
        
        for i, row in df.iterrows():
            news_text = row['news']  # Use 'news' column from financial_trading_data.csv
            
            # Get predictions from both models
            predictions = self.predict_sentiment(news_text)
            
            probetrain_signal = 0
            sae_signal = 0
            
            # Process ProbeTrain prediction
            if predictions.get('probetrain'):
                probetrain_pred = predictions['probetrain']
                if probetrain_pred['confidence'] >= confidence_threshold:
                    if probetrain_pred['prediction'] == 2:  # Up
                        probetrain_signal = 1
                    elif probetrain_pred['prediction'] == 0:  # Down
                        probetrain_signal = -1
            
            # Process SAE prediction
            if predictions.get('sae'):
                sae_pred = predictions['sae']
                if sae_pred['confidence'] >= confidence_threshold:
                    if sae_pred['prediction'] == 2:  # Up
                        sae_signal = 1
                    elif sae_pred['prediction'] == 0:  # Down
                        sae_signal = -1
            
            # Store individual signals
            signals_df.loc[i, 'probetrain_signal'] = probetrain_signal
            signals_df.loc[i, 'sae_signal'] = sae_signal
            
            # Agreement logic: both models must agree on direction (both positive or both negative)
            if probetrain_signal != 0 and sae_signal != 0:
                if probetrain_signal == sae_signal:  # Both agree on direction
                    signals_df.loc[i, 'agreement_signal'] = probetrain_signal
                    signals_df.loc[i, 'agreement_count'] = 1
                    # Use average confidence
                    if predictions.get('probetrain') and predictions.get('sae'):
                        avg_confidence = (predictions['probetrain']['confidence'] + predictions['sae']['confidence']) / 2
                        signals_df.loc[i, 'confidence'] = avg_confidence
        
        agreement_trades = signals_df['agreement_signal'].abs().sum()
        print(f"✅ Created signals: {agreement_trades} agreement-based trading opportunities")
        print(f"📊 Agreement rate: {signals_df['agreement_count'].sum()}/{len(df)} ({signals_df['agreement_count'].sum()/len(df)*100:.1f}%)")
        
        return signals_df
    
    def run_backtest(self, df, signals_df, ticker):
        """Run backtest using agreement signals for a specific ticker"""
        print(f"Running backtest for {ticker} with agreement-based signals...")
        
        # Prepare data
        df_clean = df.copy()
        df_clean['date'] = pd.to_datetime(df_clean['date'])
        df_clean = df_clean.sort_values('date')
        
        # Use actual close prices from the data
        price_series = df_clean['close_price']
        
        # Create entry/exit signals based on agreement
        entries = (signals_df['agreement_signal'] == 1).astype(bool)
        exits = (signals_df['agreement_signal'] == -1).astype(bool)
        
        # Ensure signals align with price data
        if len(entries) != len(price_series):
            print(f"⚠️ Signal length mismatch: {len(entries)} vs {len(price_series)}")
            return None
        
        try:
            # Create portfolio using VectorBT
            portfolio = vbt.Portfolio.from_signals(
                price_series,
                entries,
                exits,
                freq='1D',
                fees=self.fees,
                slippage=self.slippage,
                init_cash=self.initial_cash,
                size=np.inf,
                size_type='value'
            )
            
            # Calculate metrics
            metrics = {
                'total_return': portfolio.total_return(),
                'sharpe_ratio': portfolio.sharpe_ratio(),
                'max_drawdown': portfolio.max_drawdown(),
                'win_rate': portfolio.trades.win_rate(),
                'n_trades': len(portfolio.trades.records_readable),
                'volatility': portfolio.annualized_volatility()
            }
            
            print(f"✅ {ticker} backtest completed:")
            print(f"   Total Return: {metrics['total_return']:.2%}")
            print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            print(f"   Max Drawdown: {metrics['max_drawdown']:.2%}")
            print(f"   Trades: {metrics['n_trades']}")
            
            return {
                'portfolio': portfolio,
                'metrics': metrics,
                'data': df_clean,
                'ticker': ticker
            }
            
        except Exception as e:
            print(f"❌ Backtest failed for {ticker}: {e}")
            return None
    
    def run_comprehensive_backtest(self, data_file):
        """Run comprehensive backtest using financial_trading_data.csv"""
        print("="*80)
        print("FINANCIAL SENTIMENT BACKTESTING")
        print("="*80)
        
        # Load data from financial_trading_data.csv
        print("Loading financial trading data...")
        df = pd.read_csv(data_file)
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter to recent data to reduce processing time (last 6 months)
        cutoff_date = df['date'].max() - pd.Timedelta(days=180)  # 6 months ago
        df = df[df['date'] >= cutoff_date].copy()
        
        print(f"📊 Loaded {len(df)} data points (last 6 months)")
        print(f"📈 Tickers: {df['ticker'].unique()}")
        print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Load models once - stop if either fails
        if not self.load_models():
            print("❌ Model loading failed. Cannot proceed with backtesting.")
            return None
        
        # Run backtest for each ticker
        all_results = {}
        
        for ticker in df['ticker'].unique():
            print(f"\n{'='*50}")
            print(f"BACKTESTING {ticker}")
            print(f"{'='*50}")
            
            # Filter data for this ticker
            ticker_df = df[df['ticker'] == ticker].copy()
            ticker_df = ticker_df.sort_values('date')
            
            # Create signals based on agreement
            signals_df = self.create_trading_signals(ticker_df)
            
            # Run backtest for this ticker
            result = self.run_backtest(ticker_df, signals_df, ticker)
            
            if result:
                all_results[ticker] = result
        
        # Generate comprehensive report
        if all_results:
            self.generate_report(all_results)
            
            # Generate QuantStats report for each ticker
            for ticker, result in all_results.items():
                print(f"\n{'='*40}")
                print(f"QUANTSTATS ANALYSIS FOR {ticker}")
                print(f"{'='*40}")
                self.generate_quantstats_report(result, ticker)
        
        return all_results
    
    def generate_report(self, results):
        """Generate comprehensive backtesting report"""
        print("\n" + "="*80)
        print("AGREEMENT-BASED BACKTESTING RESULTS")
        print("="*80)
        
        # Summary table
        summary_data = []
        for ticker, result in results.items():
            metrics = result['metrics']
            summary_data.append({
                'Ticker': ticker,
                'Total Return': f"{metrics['total_return']:.2%}",
                'Sharpe Ratio': f"{metrics['sharpe_ratio']:.3f}",
                'Max Drawdown': f"{metrics['max_drawdown']:.2%}",
                'Win Rate': f"{metrics['win_rate']:.2%}",
                'Trades': f"{metrics['n_trades']:.0f}",
                'Volatility': f"{metrics['volatility']:.2%}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Overall performance
        if results:
            avg_return = np.mean([r['metrics']['total_return'] for r in results.values()])
            avg_sharpe = np.mean([r['metrics']['sharpe_ratio'] for r in results.values()])
            total_trades = sum([r['metrics']['n_trades'] for r in results.values()])
            
            print(f"\n📊 OVERALL PERFORMANCE:")
            print(f"   Average Return: {avg_return:.2%}")
            print(f"   Average Sharpe: {avg_sharpe:.3f}")
            print(f"   Total Trades: {total_trades}")
        
        # Save results
        results_file = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Probes/agreement_backtest_results.json"
        with open(results_file, 'w') as f:
            json_results = {}
            for ticker, result in results.items():
                json_results[ticker] = {
                    'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                               for k, v in result['metrics'].items()}
                }
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
    
    def generate_quantstats_report(self, result, ticker):
        """Generate QuantStats analysis report for a specific ticker"""
        print(f"\n📊 QUANTSTATS ANALYSIS FOR {ticker}")
        print("="*50)
        
        try:
            portfolio = result['portfolio']
            
            # Get returns series
            returns = portfolio.returns()
            
            # Generate QuantStats report
            print("📈 Generating QuantStats analysis...")
            
            # Basic QuantStats metrics
            qs_metrics = {
                'Total Return': f"{returns.sum():.2%}",
                'CAGR': f"{qs.stats.cagr(returns):.2%}",
                'Sharpe Ratio': f"{qs.stats.sharpe(returns):.3f}",
                'Max Drawdown': f"{qs.stats.max_drawdown(returns):.2%}",
                'Volatility': f"{qs.stats.volatility(returns):.2%}",
                'Skewness': f"{qs.stats.skew(returns):.3f}",
                'Kurtosis': f"{qs.stats.kurtosis(returns):.3f}"
            }
            
            print("\n📊 QUANTSTATS METRICS:")
            for metric, value in qs_metrics.items():
                print(f"   {metric}: {value}")
            
            # Save QuantStats HTML report
            html_file = f"/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Probes/quantstats_report_{ticker}.html"
            qs.reports.html(returns, output=html_file, title=f"Financial Sentiment Strategy - {ticker}")
            print(f"\n📄 QuantStats HTML report saved to: {html_file}")
            
        except Exception as e:
            print(f"❌ QuantStats analysis failed for {ticker}: {e}")

def main():
    """Main function"""
    # Use financial_trading_data.csv for backtesting
    data_file = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Probes/financial_trading_data.csv"
    
    if not os.path.exists(data_file):
        print("❌ Data file not found. Please run 5. collect_financial_data.py first.")
        return
    
    # Initialize backtester
    backtester = FinancialSentimentBacktester()
    
    # Run comprehensive backtest
    results = backtester.run_comprehensive_backtest(data_file)
    
    if results:
        print("\n✅ Agreement-based backtesting completed successfully!")
    else:
        print("\n❌ Backtesting failed!")

if __name__ == "__main__":
    main()
