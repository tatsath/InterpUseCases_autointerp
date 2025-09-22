"""
Data Fetcher for Crypto Market Data
Fetches OHLCV data from Binance using CCXT and saves to parquet files
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import List, Dict, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoDataFetcher:
    """Fetches crypto market data from Binance exchange"""
    
    def __init__(self, exchange_name: str = 'binanceus'):
        """Initialize the data fetcher with exchange configuration"""
        self.exchange = getattr(ccxt, exchange_name)({
            'rateLimit': 1200,  # Respect rate limits
            'enableRateLimit': True,
        })
        
        # Supported symbols and timeframes (Binance.US compatible)
        self.symbols = ['BTC/USD', 'ETH/USD', 'DOGE/USD', 'ADA/USD', 'SOL/USD']
        self.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        
    def fetch_ohlcv(self, 
                   symbol: str, 
                   timeframe: str = '1m',
                   since: Optional[datetime] = None,
                   limit: int = 1000) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single symbol
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Data timeframe ('1m', '5m', etc.)
            since: Start datetime (default: 30 days ago)
            limit: Number of candles to fetch per request
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            if since is None:
                since = datetime.now() - timedelta(days=30)
            
            since_timestamp = int(since.timestamp() * 1000)
            
            logger.info(f"Fetching {symbol} {timeframe} data since {since}")
            
            ohlcv = self.exchange.fetch_ohlcv(
                symbol, 
                timeframe, 
                since=since_timestamp, 
                limit=limit
            )
            
            if not ohlcv:
                logger.warning(f"No data received for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df = df.set_index('timestamp')
            df['symbol'] = symbol.replace('/', '')
            
            # Add basic price features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            logger.info(f"Fetched {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_multiple_symbols(self, 
                             symbols: List[str], 
                             timeframe: str = '1m',
                             since: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch OHLCV data for multiple symbols
        
        Args:
            symbols: List of trading pair symbols
            timeframe: Data timeframe
            since: Start datetime
            
        Returns:
            Combined DataFrame with data for all symbols
        """
        all_data = []
        
        for symbol in symbols:
            logger.info(f"Fetching data for {symbol}")
            df = self.fetch_ohlcv(symbol, timeframe, since)
            if not df.empty:
                all_data.append(df)
        
        if not all_data:
            logger.warning("No data fetched for any symbol")
            return pd.DataFrame()
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=False)
        combined_df = combined_df.sort_index()
        
        logger.info(f"Combined data: {len(combined_df)} total candles for {len(symbols)} symbols")
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
    
    def get_historical_data(self, 
                          symbols: List[str] = None,
                          timeframe: str = '1m',
                          days_back: int = 30,
                          save_file: str = None) -> pd.DataFrame:
        """
        Get historical data and optionally save it
        
        Args:
            symbols: List of symbols to fetch (default: BTC/USDT, DOGE/USDT)
            timeframe: Data timeframe
            days_back: Number of days to go back
            save_file: Filename to save data (optional)
            
        Returns:
            DataFrame with historical data
        """
        if symbols is None:
            symbols = ['BTC/USDT', 'DOGE/USDT']
        
        since = datetime.now() - timedelta(days=days_back)
        
        df = self.fetch_multiple_symbols(symbols, timeframe, since)
        
        if save_file and not df.empty:
            self.save_to_parquet(df, save_file)
        
        return df

def main():
    """Example usage of the data fetcher"""
    fetcher = CryptoDataFetcher()
    
    # Fetch recent data
    df = fetcher.get_historical_data(
        symbols=['BTC/USDT', 'DOGE/USDT'],
        timeframe='1m',
        days_back=7,
        save_file='crypto_data_1m.parquet'
    )
    
    if not df.empty:
        print(f"Data shape: {df.shape}")
        print(f"Symbols: {df['symbol'].unique()}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print("\nSample data:")
        print(df.head())
        
        # Basic statistics
        print("\nPrice statistics:")
        print(df.groupby('symbol')['close'].describe())

if __name__ == "__main__":
    main()
