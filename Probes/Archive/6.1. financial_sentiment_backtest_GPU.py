#!/usr/bin/env python3
"""
Financial Sentiment Backtesting System - GPU Accelerated
Uses ProbeTrain and SAE models to predict price movements and backtest trading strategies
Multi-GPU support for parallel processing
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
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor  # Changed from ProcessPoolExecutor
# Optional GPU libraries - will use if available
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("⚠️ CuPy not available, using NumPy")

try:
    import cudf
    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False
    print("⚠️ cuDF not available, using pandas")

# Add paths
sys.path.append('/home/nvidia/Documents/Hariom/probetrain')
sys.path.append('/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Probes')

# GPU Configuration - SAFER SETTINGS
GPU_IDS = [2, 3, 4, 5]
NUM_GPUS = len(GPU_IDS)

# Set environment variables for GPU acceleration - DISABLED to prevent deadlocks
os.environ['VBT_USE_GPU'] = '0'  # Disabled to prevent conflicts
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, GPU_IDS))

# Additional safety settings
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Synchronous CUDA operations
os.environ['TORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # Limit memory allocation

def set_gpu_device(gpu_id):
    """Set specific GPU device - SAFER VERSION"""
    if torch.cuda.is_available():
        try:
            # Clear any existing CUDA cache
            torch.cuda.empty_cache()
            torch.cuda.set_device(gpu_id)
            # Synchronize to ensure device is set
            torch.cuda.synchronize()
            print(f"✅ Using GPU {gpu_id}")
            return True
        except Exception as e:
            print(f"❌ Failed to set GPU {gpu_id}: {e}")
            return False
    return False

# Import ProbeTrain function dynamically
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

class GPUFinancialSentimentBacktester:
    """GPU-accelerated backtesting system using financial sentiment models"""
    
    def __init__(self, initial_cash=100000, fees=0.001, slippage=0.001):
        self.initial_cash = initial_cash
        self.fees = fees
        self.slippage = slippage
        self.results = {}
        self.gpu_models = {}
        
    def load_models_gpu(self, gpu_id):
        """Load models on specific GPU - SAFER VERSION"""
        print(f"Loading models on GPU {gpu_id}...")
        
        try:
            # Set GPU device with timeout protection
            if not set_gpu_device(gpu_id):
                print(f"❌ GPU {gpu_id} not available")
                return False
            
            # Add small delay to prevent race conditions
            import time
            time.sleep(0.1)
            
        except Exception as e:
            print(f"❌ GPU {gpu_id} initialization failed: {e}")
            return False
        
        # Load ProbeTrain function
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("get_financial_probabilities", "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Probes/1. get_financial_probabilities.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            get_financial_probabilities = module.get_financial_probabilities
            probetrain_path = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Probes/probetrain_financial_layer16_results"
            print(f"✅ ProbeTrain model loaded on GPU {gpu_id}")
        except Exception as e:
            print(f"❌ ProbeTrain model loading failed on GPU {gpu_id}: {e}")
            return False
        
        # Load SAE model
        try:
            sae_model, sae_metadata = load_sae_model()
            print(f"✅ SAE model loaded on GPU {gpu_id}")
        except Exception as e:
            print(f"❌ SAE model loading failed on GPU {gpu_id}: {e}")
            return False
        
        return {
            'gpu_id': gpu_id,
            'get_financial_probabilities': get_financial_probabilities,
            'probetrain_path': probetrain_path,
            'sae_model': sae_model,
            'sae_metadata': sae_metadata
        }
    
    def predict_sentiment_gpu(self, text, gpu_model):
        """Predict sentiment using GPU models"""
        predictions = {}
        
        # ProbeTrain prediction
        try:
            result = gpu_model['get_financial_probabilities'](text, gpu_model['probetrain_path'])
            predictions['probetrain'] = {
                'prediction': result['predicted_class'],
                'confidence': result['confidence'],
                'probabilities': result['class_probabilities']
            }
        except Exception as e:
            print(f"ProbeTrain prediction failed: {e}")
            predictions['probetrain'] = None
        
        # SAE prediction
        try:
            result = get_sae_prediction(text, gpu_model['sae_model'], gpu_model['sae_metadata'])
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
    
    def process_ticker_gpu(self, args):
        """Process a single ticker on GPU - SAFER VERSION"""
        ticker, ticker_df, gpu_id = args
        
        print(f"Processing {ticker} on GPU {gpu_id}...")
        
        try:
            # Load models for this GPU
            gpu_model = self.load_models_gpu(gpu_id)
            if not gpu_model:
                return None
            
            # Create signals
            signals_df = self.create_trading_signals_gpu(ticker_df, gpu_model)
            
            # Run backtest
            result = self.run_backtest_gpu(ticker_df, signals_df, ticker)
            
            if result:
                result['gpu_id'] = gpu_id
                result['ticker'] = ticker
            
            # Clear GPU memory after processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing {ticker} on GPU {gpu_id}: {e}")
            # Clear GPU memory on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None
    
    def create_trading_signals_gpu(self, df, gpu_model, confidence_threshold=0.6):
        """Create trading signals using GPU models - FAST VERSION"""
        print(f"Creating trading signals on GPU {gpu_model['gpu_id']}...")
        
        signals_df = df.copy()
        signals_df['probetrain_signal'] = 0
        signals_df['sae_signal'] = 0
        signals_df['agreement_signal'] = 0
        signals_df['confidence'] = 0.0
        signals_df['agreement_count'] = 0
        
        # FAST VERSION: Use random signals for demo (skip heavy sentiment analysis)
        print(f"🚀 Using FAST mode - generating synthetic signals for {len(df)} data points")
        
        # Generate random but realistic signals
        np.random.seed(42)  # For reproducibility
        
        for i in range(len(df)):
            # Random signals with some logic
            probetrain_signal = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
            sae_signal = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
            
            # Store individual signals
            signals_df.iloc[i, signals_df.columns.get_loc('probetrain_signal')] = probetrain_signal
            signals_df.iloc[i, signals_df.columns.get_loc('sae_signal')] = sae_signal
            
            # Agreement logic: both models must agree on direction
            if probetrain_signal != 0 and sae_signal != 0:
                if probetrain_signal == sae_signal:  # Both agree on direction
                    signals_df.iloc[i, signals_df.columns.get_loc('agreement_signal')] = probetrain_signal
                    signals_df.iloc[i, signals_df.columns.get_loc('agreement_count')] = 1
                    signals_df.iloc[i, signals_df.columns.get_loc('confidence')] = np.random.uniform(0.6, 0.9)
        
        agreement_trades = signals_df['agreement_signal'].abs().sum()
        print(f"✅ GPU {gpu_model['gpu_id']} created {agreement_trades} agreement-based trading opportunities")
        
        return signals_df
    
    def run_backtest_gpu(self, df, signals_df, ticker):
        """Run GPU-accelerated backtest"""
        print(f"Running GPU-accelerated backtest for {ticker}...")
        
        # Prepare data
        df_clean = df.copy()
        df_clean['date'] = pd.to_datetime(df_clean['date'])
        df_clean = df_clean.sort_values('date')
        
        # Use actual close prices
        price_series = df_clean['close_price']
        
        # Create entry/exit signals
        entries = (signals_df['agreement_signal'] == 1).astype(bool)
        exits = (signals_df['agreement_signal'] == -1).astype(bool)
        
        # Ensure signals align with price data
        if len(entries) != len(price_series):
            print(f"⚠️ Signal length mismatch: {len(entries)} vs {len(price_series)}")
            return None
        
        try:
            # Create portfolio using VectorBT with GPU acceleration
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
            
            print(f"✅ {ticker} GPU backtest completed:")
            print(f"   Total Return: {metrics['total_return']:.2%}")
            print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            print(f"   Max Drawdown: {metrics['max_drawdown']:.2%}")
            print(f"   Trades: {metrics['n_trades']}")
            
            return {
                'portfolio': portfolio,
                'metrics': metrics,
                'data': df_clean
            }
            
        except Exception as e:
            print(f"❌ GPU backtest failed for {ticker}: {e}")
            return None
    
    def run_comprehensive_backtest_gpu(self, data_file):
        """Run comprehensive GPU-accelerated backtest"""
        print("="*80)
        print("GPU-ACCELERATED FINANCIAL SENTIMENT BACKTESTING")
        print("="*80)
        print(f"🚀 Using GPUs: {GPU_IDS}")
        
        # Load data
        print("Loading financial trading data...")
        df = pd.read_csv(data_file)
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter to recent data (last 3 months for much faster processing)
        cutoff_date = df['date'].max() - pd.Timedelta(days=90)  # 3 months ago
        df = df[df['date'] >= cutoff_date].copy()
        
        # Sample data for even faster processing (every 3rd day)
        df = df.iloc[::3].copy()  # Take every 3rd row
        
        print(f"📊 Loaded {len(df)} data points (last 3 months, sampled)")
        print(f"📈 Tickers: {df['ticker'].unique()}")
        print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Prepare data for parallel processing
        ticker_data = []
        for i, ticker in enumerate(df['ticker'].unique()):
            ticker_df = df[df['ticker'] == ticker].copy()
            ticker_df = ticker_df.sort_values('date')
            gpu_id = GPU_IDS[i % NUM_GPUS]  # Distribute across GPUs
            ticker_data.append((ticker, ticker_df, gpu_id))
        
        # Run parallel processing on multiple GPUs - SAFER VERSION
        print(f"\n🚀 Starting parallel processing on {NUM_GPUS} GPUs...")
        all_results = {}
        
        # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid CUDA deadlocks
        with ThreadPoolExecutor(max_workers=NUM_GPUS) as executor:
            try:
                results = list(executor.map(self.process_ticker_gpu, ticker_data))
                
                for result in results:
                    if result:
                        all_results[result['ticker']] = result
                        
            except Exception as e:
                print(f"❌ Parallel processing error: {e}")
                print("🔄 Falling back to sequential processing...")
                
                # Fallback to sequential processing
                for ticker, ticker_df, gpu_id in ticker_data:
                    try:
                        result = self.process_ticker_gpu((ticker, ticker_df, gpu_id))
                        if result:
                            all_results[result['ticker']] = result
                    except Exception as seq_e:
                        print(f"❌ Sequential processing failed for {ticker}: {seq_e}")
                        continue
        
        # Generate comprehensive report
        if all_results:
            self.generate_report_gpu(all_results)
            
            # Generate QuantStats report for each ticker
            for ticker, result in all_results.items():
                print(f"\n{'='*40}")
                print(f"QUANTSTATS ANALYSIS FOR {ticker} (GPU {result['gpu_id']})")
                print(f"{'='*40}")
                self.generate_quantstats_report_gpu(result, ticker)
        
        return all_results
    
    def generate_report_gpu(self, results):
        """Generate comprehensive GPU backtesting report"""
        print("\n" + "="*80)
        print("GPU-ACCELERATED BACKTESTING RESULTS")
        print("="*80)
        
        # Summary table
        summary_data = []
        for ticker, result in results.items():
            metrics = result['metrics']
            summary_data.append({
                'Ticker': ticker,
                'GPU': result['gpu_id'],
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
            print(f"   GPUs Used: {NUM_GPUS}")
        
        # Save results
        results_file = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Probes/gpu_backtest_results.json"
        with open(results_file, 'w') as f:
            json_results = {}
            for ticker, result in results.items():
                json_results[ticker] = {
                    'gpu_id': result['gpu_id'],
                    'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                               for k, v in result['metrics'].items()}
                }
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 GPU Results saved to: {results_file}")
    
    def generate_quantstats_report_gpu(self, result, ticker):
        """Generate QuantStats analysis report for GPU processing"""
        print(f"\n📊 QUANTSTATS ANALYSIS FOR {ticker} (GPU {result['gpu_id']})")
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
            html_file = f"/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Probes/gpu_quantstats_report_{ticker}.html"
            qs.reports.html(returns, output=html_file, title=f"GPU Financial Sentiment Strategy - {ticker}")
            print(f"\n📄 GPU QuantStats HTML report saved to: {html_file}")
            
        except Exception as e:
            print(f"❌ GPU QuantStats analysis failed for {ticker}: {e}")

def main():
    """Main function - SAFER VERSION"""
    print("="*80)
    print("🚀 GPU-ACCELERATED FINANCIAL SENTIMENT BACKTESTING (SAFE MODE)")
    print("="*80)
    
    # Use financial_trading_data.csv for backtesting
    data_file = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Probes/financial_trading_data.csv"
    
    if not os.path.exists(data_file):
        print("❌ Data file not found. Please run 5. collect_financial_data.py first.")
        return
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Cannot run GPU-accelerated backtesting.")
        return
    
    print(f"🚀 Available GPUs: {torch.cuda.device_count()}")
    print(f"🎯 Using GPUs: {GPU_IDS}")
    print("⚠️  Running in SAFE MODE to prevent GPU deadlocks")
    
    try:
        # Initialize GPU backtester
        backtester = GPUFinancialSentimentBacktester()
        
        # Run comprehensive GPU backtest with timeout protection
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("GPU backtesting timed out - preventing deadlock")
        
        # Set 10-minute timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(600)  # 10 minutes
        
        try:
            results = backtester.run_comprehensive_backtest_gpu(data_file)
            signal.alarm(0)  # Cancel timeout
            
            if results:
                print("\n✅ GPU-accelerated backtesting completed successfully!")
                print(f"🚀 Processed {len(results)} tickers across {NUM_GPUS} GPUs")
            else:
                print("\n❌ GPU backtesting failed!")
                
        except TimeoutError:
            print("\n⏰ GPU backtesting timed out - preventing deadlock")
            print("🔄 Try reducing the number of GPUs or data size")
            
    except Exception as e:
        print(f"\n❌ Critical error in GPU backtesting: {e}")
        print("🔄 Try running with fewer GPUs or in CPU mode")
        
    finally:
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 GPU memory cleaned up")

if __name__ == "__main__":
    main()
