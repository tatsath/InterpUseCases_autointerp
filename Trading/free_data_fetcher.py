"""
Free Crypto Data Fetcher
Uses free APIs that don't require authentication
"""

import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
import time
import logging
from typing import List, Dict, Optional
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FreeCryptoDataFetcher:
    """Fetches crypto data from free APIs without authentication"""
    
    def __init__(self):
        """Initialize the free data fetcher"""
        self.symbols = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum', 
            'DOGE': 'dogecoin',
            'ADA': 'cardano',
            'SOL': 'solana'
        }
        
        # Yahoo Finance symbols
        self.yahoo_symbols = {
            'BTC': 'BTC-USD',
            'ETH': 'ETH-USD',
            'DOGE': 'DOGE-USD',
            'ADA': 'ADA-USD',
            'SOL': 'SOL-USD'
        }
    
    def fetch_from_yahoo_finance(self, 
                                symbol: str, 
                                period: str = '30d',
                                interval: str = '1m') -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance using yfinance
        
        Args:
            symbol: Crypto symbol (BTC, ETH, DOGE, etc.)
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            if symbol not in self.yahoo_symbols:
                logger.error(f"Symbol {symbol} not supported")
                return pd.DataFrame()
            
            yahoo_symbol = self.yahoo_symbols[symbol]
            logger.info(f"Fetching {symbol} data from Yahoo Finance...")
            
            # Fetch data
            ticker = yf.Ticker(yahoo_symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data received for {symbol}")
                return pd.DataFrame()
            
            # Clean and format data
            df = data.copy()
            df.columns = df.columns.str.lower()
            df = df.rename(columns={'adj close': 'close'})
            
            # Remove unwanted columns
            unwanted_cols = ['dividends', 'stock splits']
            df = df.drop(columns=[col for col in unwanted_cols if col in df.columns])
            
            # Add required columns
            df['symbol'] = symbol
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Ensure we have all OHLCV columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    if col == 'volume':
                        df[col] = 1000000  # Default volume
                    else:
                        df[col] = df['close']  # Use close price as default
            
            logger.info(f"Fetched {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {symbol} from Yahoo Finance: {str(e)}")
            return pd.DataFrame()
    
    def fetch_from_coingecko(self, 
                            symbol: str, 
                            days: int = 30) -> pd.DataFrame:
        """
        Fetch data from CoinGecko API
        
        Args:
            symbol: Crypto symbol (BTC, ETH, DOGE, etc.)
            days: Number of days of data
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            if symbol not in self.symbols:
                logger.error(f"Symbol {symbol} not supported")
                return pd.DataFrame()
            
            coin_id = self.symbols[symbol]
            logger.info(f"Fetching {symbol} data from CoinGecko...")
            
            # CoinGecko API endpoint
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'hourly' if days <= 90 else 'daily'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract price data
            prices = data['prices']
            volumes = data['total_volumes']
            
            if not prices:
                logger.warning(f"No data received for {symbol}")
                return pd.DataFrame()
            
            # Create DataFrame
            df = pd.DataFrame(prices, columns=['timestamp', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df = df.set_index('timestamp')
            
            # Add volume data
            if volumes:
                volume_df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
                volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms', utc=True)
                volume_df = volume_df.set_index('timestamp')
                df['volume'] = volume_df['volume']
            else:
                df['volume'] = 1000000  # Default volume
            
            # Create OHLC from close prices (simplified)
            df['open'] = df['close'].shift(1).fillna(df['close'])
            df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.02, len(df)))
            df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.02, len(df)))
            
            # Add required columns
            df['symbol'] = symbol
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            logger.info(f"Fetched {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {symbol} from CoinGecko: {str(e)}")
            return pd.DataFrame()
    
    def fetch_multiple_symbols(self, 
                             symbols: List[str], 
                             source: str = 'yahoo',
                             **kwargs) -> pd.DataFrame:
        """
        Fetch data for multiple symbols
        
        Args:
            symbols: List of crypto symbols
            source: Data source ('yahoo' or 'coingecko')
            **kwargs: Additional parameters
            
        Returns:
            Combined DataFrame with data for all symbols
        """
        all_data = []
        
        for symbol in symbols:
            logger.info(f"Fetching data for {symbol}...")
            
            if source == 'yahoo':
                df = self.fetch_from_yahoo_finance(symbol, **kwargs)
            elif source == 'coingecko':
                df = self.fetch_from_coingecko(symbol, **kwargs)
            else:
                logger.error(f"Unknown source: {source}")
                continue
            
            if not df.empty:
                all_data.append(df)
            else:
                logger.warning(f"No data fetched for {symbol}")
        
        if not all_data:
            logger.warning("No data fetched for any symbol")
            return pd.DataFrame()
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=False)
        combined_df = combined_df.sort_index()
        
        logger.info(f"Combined data: {len(combined_df)} total candles for {len(symbols)} symbols")
        return combined_df
    
    def create_synthetic_data(self, 
                            symbols: List[str] = None,
                            days: int = 30,
                            timeframe: str = '1min') -> pd.DataFrame:
        """
        Create synthetic crypto data for testing
        
        Args:
            symbols: List of crypto symbols
            days: Number of days of data
            timeframe: Data frequency
            
        Returns:
            DataFrame with synthetic OHLCV data
        """
        if symbols is None:
            symbols = ['BTC', 'ETH', 'DOGE']
        
        # Calculate number of periods
        if timeframe == '1min':
            periods = days * 24 * 60
        elif timeframe == '5min':
            periods = days * 24 * 12
        elif timeframe == '1h':
            periods = days * 24
        else:
            periods = days * 24 * 60
        
        all_data = []
        
        for symbol in symbols:
            logger.info(f"Creating synthetic data for {symbol}...")
            
            # Base prices
            base_prices = {
                'BTC': 50000,
                'ETH': 3000,
                'DOGE': 0.08,
                'ADA': 0.5,
                'SOL': 100
            }
            
            base_price = base_prices.get(symbol, 100)
            volatility = 0.02 if symbol in ['BTC', 'ETH'] else 0.05
            
            # Generate realistic price movements
            np.random.seed(hash(symbol) % 2**32)
            returns = np.random.normal(0.0001, volatility, periods)
            
            # Add market cycles
            cycle_period = periods // 7
            cycle_effect = 0.001 * np.sin(2 * np.pi * np.arange(periods) / cycle_period)
            returns += cycle_effect
            
            # Calculate prices
            prices = base_price * np.exp(np.cumsum(returns))
            
            # Create dates
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=days),
                periods=periods,
                freq=timeframe
            )
            
            # Generate OHLCV
            df = pd.DataFrame({
                'open': prices * (1 + np.random.normal(0, 0.001, periods)),
                'high': prices * (1 + np.random.exponential(0.005, periods)),
                'low': prices * (1 - np.random.exponential(0.005, periods)),
                'close': prices,
                'volume': np.random.exponential(1000, periods) * (1 + np.abs(returns) * 10)
            }, index=dates)
            
            # Ensure high >= max(open, close) and low <= min(open, close)
            df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
            df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
            
            # Add required columns
            df['symbol'] = symbol
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            all_data.append(df)
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=False)
        combined_df = combined_df.sort_index()
        
        logger.info(f"Created synthetic data: {len(combined_df)} candles for {len(symbols)} symbols")
        return combined_df
    
    def save_to_parquet(self, df: pd.DataFrame, filename: str, data_dir: str = 'data'):
        """Save DataFrame to parquet file"""
        os.makedirs(data_dir, exist_ok=True)
        filepath = os.path.join(data_dir, filename)
        df.to_parquet(filepath)
        logger.info(f"Data saved to {filepath}")
    
    def load_from_parquet(self, filename: str, data_dir: str = 'data') -> pd.DataFrame:
        """Load DataFrame from parquet file"""
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_parquet(filepath)
            logger.info(f"Data loaded from {filepath}")
            return df
        else:
            logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()

def main():
    """Test the free data fetcher"""
    fetcher = FreeCryptoDataFetcher()
    
    print("Testing Free Crypto Data Fetcher")
    print("=" * 50)
    
    # Test 1: Yahoo Finance
    print("\n1. Testing Yahoo Finance...")
    try:
        df_yahoo = fetcher.fetch_from_yahoo_finance('BTC', period='7d', interval='1h')
        if not df_yahoo.empty:
            print(f"   ✅ Yahoo Finance: {len(df_yahoo)} candles")
            print(f"   Latest BTC price: ${df_yahoo['close'].iloc[-1]:,.2f}")
        else:
            print("   ❌ Yahoo Finance failed")
    except Exception as e:
        print(f"   ❌ Yahoo Finance error: {str(e)}")
    
    # Test 2: CoinGecko
    print("\n2. Testing CoinGecko...")
    try:
        df_coingecko = fetcher.fetch_from_coingecko('BTC', days=7)
        if not df_coingecko.empty:
            print(f"   ✅ CoinGecko: {len(df_coingecko)} candles")
            print(f"   Latest BTC price: ${df_coingecko['close'].iloc[-1]:,.2f}")
        else:
            print("   ❌ CoinGecko failed")
    except Exception as e:
        print(f"   ❌ CoinGecko error: {str(e)}")
    
    # Test 3: Synthetic data
    print("\n3. Testing synthetic data...")
    df_synthetic = fetcher.create_synthetic_data(['BTC', 'ETH'], days=7)
    if not df_synthetic.empty:
        print(f"   ✅ Synthetic: {len(df_synthetic)} candles")
        print(f"   Symbols: {df_synthetic['symbol'].unique()}")
    
    print("\n" + "=" * 50)
    print("Free data fetcher test completed!")

if __name__ == "__main__":
    main()