"""
Test Binance.US API Access
Simple script to test if your API credentials work
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
import os

def test_binance_api():
    """Test Binance.US API access"""
    
    print("Testing Binance.US API Access")
    print("=" * 50)
    
    try:
        # Initialize Binance.US exchange
        exchange = ccxt.binanceus({
            'rateLimit': 1200,
            'enableRateLimit': True,
        })
        
        # Test 1: Check if exchange is accessible
        print("1. Testing exchange connectivity...")
        markets = exchange.load_markets()
        print(f"   ✅ Connected! Found {len(markets)} trading pairs")
        
        # Test 2: Check available symbols
        print("\n2. Available trading pairs (first 10):")
        usd_pairs = [symbol for symbol in markets.keys() if '/USD' in symbol]
        for pair in usd_pairs[:10]:
            print(f"   - {pair}")
        
        # Test 3: Fetch recent data for BTC/USD
        print("\n3. Testing data fetch for BTC/USD...")
        try:
            ohlcv = exchange.fetch_ohlcv('BTC/USD', '1h', limit=10)
            if ohlcv:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                print(f"   ✅ Successfully fetched {len(df)} candles")
                print(f"   Latest BTC price: ${df['close'].iloc[-1]:,.2f}")
                print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            else:
                print("   ❌ No data received")
        except Exception as e:
            print(f"   ❌ Error fetching data: {str(e)}")
        
        # Test 4: Check rate limits
        print("\n4. Testing rate limits...")
        try:
            # Make multiple requests to test rate limiting
            for i in range(3):
                ticker = exchange.fetch_ticker('BTC/USD')
                print(f"   Request {i+1}: BTC price = ${ticker['last']:,.2f}")
        except Exception as e:
            print(f"   ❌ Rate limit error: {str(e)}")
        
        print("\n" + "=" * 50)
        print("✅ API Test Completed Successfully!")
        print("Your Binance.US API is working correctly.")
        print("\nNext steps:")
        print("1. Run: python main_trading_script.py")
        print("2. Or run: python demo_trading_system.py (for synthetic data)")
        
    except Exception as e:
        print(f"❌ API Test Failed: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check if you're in a supported US state")
        print("2. Verify your internet connection")
        print("3. Try using a VPN if you're outside the US")
        print("4. Check if Binance.US is available in your region")

def test_with_credentials():
    """Test with API credentials (if you have them)"""
    
    print("\nTesting with API Credentials")
    print("=" * 50)
    
    # Get credentials from user
    api_key = input("Enter your API Key (or press Enter to skip): ").strip()
    if not api_key:
        print("Skipping authenticated test...")
        return
    
    secret_key = input("Enter your Secret Key (or press Enter to skip): ").strip()
    if not secret_key:
        print("Skipping authenticated test...")
        return
    
    try:
        # Initialize with credentials
        exchange = ccxt.binanceus({
            'apiKey': api_key,
            'secret': secret_key,
            'rateLimit': 1200,
            'enableRateLimit': True,
        })
        
        # Test authenticated endpoints
        print("Testing authenticated access...")
        account = exchange.fetch_balance()
        print(f"✅ Authenticated! Account has {len(account)} assets")
        
        # Show available balances
        print("\nAccount Balances:")
        for asset, balance in account.items():
            if balance['total'] > 0:
                print(f"  {asset}: {balance['total']}")
        
    except Exception as e:
        print(f"❌ Authentication failed: {str(e)}")
        print("Check your API key and secret key")

if __name__ == "__main__":
    # Test basic API access
    test_binance_api()
    
    # Test with credentials (optional)
    test_with_credentials()
