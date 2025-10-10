#!/usr/bin/env python3
"""
Financial Data Collector
Collects historical data for Tesla, Boeing, and Bitcoin using Hugging Face datasets
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
import requests

def get_data(symbol, start_date, end_date):
    """Get data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        return data
    except Exception as e:
        print(f"Error getting data for {symbol}: {e}")
        return None

def find_top_tickers_in_huggingface():
    """Find the top 2 tickers with most news articles in Hugging Face dataset"""
    try:
        print(f"📰 Loading Hugging Face SP500 Financial News dataset to find top tickers...")
        
        # Try to load using datasets library first
        try:
            from datasets import load_dataset
            dataset = load_dataset("KrossKinetic/SP500-Financial-News-Articles-Time-Series", split="train")
            df = dataset.to_pandas()
            print(f"📊 Loaded {len(df)} total news articles from Hugging Face")
            
            # Find tickers with most news articles
            ticker_counts = df['symbol'].value_counts()
            print(f"📊 All tickers by news count:")
            for i, (ticker, count) in enumerate(ticker_counts.items()):
                print(f"  {i+1}. {ticker}: {count} articles")
            
            # Find tickers with at least 50+ articles (more comprehensive coverage)
            comprehensive_tickers = ticker_counts[ticker_counts >= 50]
            print(f"\n📈 Tickers with 50+ articles: {len(comprehensive_tickers)}")
            for ticker, count in comprehensive_tickers.items():
                print(f"  {ticker}: {count} articles")
            
            # If we have tickers with 50+ articles, use the top 2
            if len(comprehensive_tickers) >= 2:
                top_tickers = comprehensive_tickers.head(2).index.tolist()
                print(f"🎯 Selected top 2 comprehensive tickers: {top_tickers}")
            else:
                # Fall back to top 2 overall
                top_tickers = ticker_counts.head(2).index.tolist()
                print(f"🎯 Selected top 2 overall tickers: {top_tickers}")
            
            return top_tickers
                
        except ImportError:
            print("⚠️ datasets library not available, falling back to TSLA and BA...")
            return ['TSLA', 'BA']
            
    except Exception as e:
        print(f"❌ Error finding top tickers: {e}")
        return ['TSLA', 'BA']

def get_historical_news_from_huggingface(symbol, start_date, end_date):
    """Get historical financial news from Hugging Face datasets using API"""
    try:
        print(f"📰 Loading Hugging Face SP500 Financial News dataset for {symbol}...")
        
        # Try to load using datasets library first
        try:
            from datasets import load_dataset
            dataset = load_dataset("KrossKinetic/SP500-Financial-News-Articles-Time-Series", split="train")
            df = dataset.to_pandas()
            print(f"📊 Loaded {len(df)} total news articles from Hugging Face")
            
        except ImportError:
            print("⚠️ datasets library not available, trying alternative approach...")
            return get_yahoo_news_fallback(symbol, start_date, end_date)
        
        # Filter for our symbol (case-insensitive)
        symbol_news = df[df['symbol'].str.upper() == symbol.upper()].copy()
        print(f"📈 Found {len(symbol_news)} news articles for {symbol}")
        
        if len(symbol_news) == 0:
            print(f"⚠️ No news found for {symbol} in Hugging Face dataset")
            return []
        
        # Check what date column is available
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        print(f"📅 Available date columns: {date_columns}")
        
        # Use the first available date column
        date_col = date_columns[0] if date_columns else 'date'
        print(f"📅 Using date column: {date_col}")
        
        # Convert date column to datetime
        symbol_news[date_col] = pd.to_datetime(symbol_news[date_col])
        
        # Filter by date range
        start_date_obj = pd.to_datetime(start_date)
        end_date_obj = pd.to_datetime(end_date)
        
        filtered_news = symbol_news[
            (symbol_news[date_col] >= start_date_obj) & 
            (symbol_news[date_col] <= end_date_obj)
        ]
        
        print(f"📅 Filtered to {len(filtered_news)} news articles within date range for {symbol}")
        
        # Convert to list of dictionaries
        news_list = []
        for _, row in filtered_news.iterrows():
            news_list.append({
                'date': row[date_col],
                'title': row['Title'],  # Use 'Title' column
                'content': row['Text'] if 'Text' in row else row['Title'],  # Use 'Text' column
                'symbol': row['symbol']
            })
        
        return news_list
        
    except Exception as e:
        print(f"❌ Error loading Hugging Face dataset for {symbol}: {e}")
        print("🔄 Falling back to Yahoo Finance news...")
        return get_yahoo_news_fallback(symbol, start_date, end_date)

def get_yahoo_news_fallback(symbol, start_date, end_date):
    """Fallback to Yahoo Finance news (limited)"""
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        print(f"📰 Found {len(news)} recent news articles for {symbol} (Yahoo Finance)")
        
        # Convert to our format
        news_list = []
        for article in news:
            try:
                news_date_str = article['content']['pubDate']
                news_date = datetime.fromisoformat(news_date_str.replace('Z', '+00:00'))
                
                news_list.append({
                    'date': news_date,
                    'title': article['content']['title'],
                    'content': article['content']['title'],
                    'symbol': symbol
                })
            except (KeyError, ValueError):
                continue
        
        return news_list
        
    except Exception as e:
        print(f"❌ Error getting Yahoo Finance news for {symbol}: {e}")
        return []

def create_news_data(symbol, dates, prices, start_date, end_date):
    """Create news data using Hugging Face historical news - last news of each trading day"""
    news_data = []
    
    # Get historical news from Hugging Face (with Yahoo Finance fallback)
    historical_news = get_historical_news_from_huggingface(symbol, start_date, end_date)
    
    if not historical_news:
        print(f"⚠️ No news found for {symbol}, skipping...")
        return news_data
    
    print(f"📰 Found {len(historical_news)} historical news articles for {symbol}")
    
    # Group news by trading day and get the last news of each day
    news_by_date = {}
    
    for news_item in historical_news:
        try:
            # Extract news headline and date
            news_headline = news_item['title']
            news_date = news_item['date']
            
            # Find the closest trading date to the news date
            closest_trading_date = None
            closest_price = None
            
            # Convert news date to string for comparison
            news_date_str_formatted = news_date.strftime('%Y-%m-%d')
            
            # Find the closest date in our price data
            for j, (date, price) in enumerate(zip(dates, prices)):
                date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
                if date_str == news_date_str_formatted:
                    closest_trading_date = date
                    closest_price = price
                    break
            
            # If exact date not found, find the closest date within 1 day
            if closest_trading_date is None:
                news_date_obj = news_date.date()
                
                for j, (date, price) in enumerate(zip(dates, prices)):
                    date_obj = date.date() if hasattr(date, 'date') else datetime.strptime(str(date), '%Y-%m-%d').date()
                    date_diff = abs((date_obj - news_date_obj).days)
                    
                    if date_diff <= 1:  # Within 1 day
                        closest_trading_date = date
                        closest_price = price
                        break
            
            # Only add if we found a matching trading date
            if closest_trading_date is not None and closest_price is not None:
                date_key = closest_trading_date.strftime('%Y-%m-%d')
                
                # Store the latest news for each trading day
                if date_key not in news_by_date:
                    news_by_date[date_key] = {
                        'date': closest_trading_date,
                        'ticker': symbol,
                        'news': news_headline,
                        'close_price': closest_price,
                        'news_time': news_date
                    }
        else:
                    # If we already have news for this date, keep the latest one
                    if news_date > news_by_date[date_key]['news_time']:
                        news_by_date[date_key] = {
                            'date': closest_trading_date,
                            'ticker': symbol,
                            'news': news_headline,
                            'close_price': closest_price,
                            'news_time': news_date
                        }
                        
        except (KeyError, TypeError, ValueError) as e:
            print(f"⚠️ Error processing news item: {e}")
            continue
    
    # Convert to list and sort by date
    for date_key in sorted(news_by_date.keys()):
        item = news_by_date[date_key]
        news_data.append({
            'date': item['date'],
            'ticker': item['ticker'],
            'news': item['news'],
            'close_price': item['close_price']
        })
    
    print(f"📊 Successfully matched {len(news_data)} unique trading days with news for {symbol}")
    return news_data


def create_comprehensive_news_data_with_minimum(symbol, dates, prices, start_date, end_date, min_days=100):
    """Create comprehensive news data ensuring minimum coverage"""
    print(f"🔍 Creating comprehensive news data for {symbol} (targeting {min_days}+ days)...")
    
    # Get news from Hugging Face
    huggingface_news = get_historical_news_from_huggingface(symbol, start_date, end_date)
    
    # Get news from Yahoo Finance (recent)
    yahoo_news = get_yahoo_news_fallback(symbol, start_date, end_date)
    
    # Combine all real news sources
    all_news = huggingface_news + yahoo_news
    
    print(f"📰 Real news articles found: {len(all_news)} (HF: {len(huggingface_news)}, Yahoo: {len(yahoo_news)})")
    
    # Group real news by trading day
    real_news_by_date = {}
    
    for news_item in all_news:
        try:
            # Extract news headline and date
            news_headline = news_item['title']
            news_date = news_item['date']
            
            # Find the closest trading date to the news date
            closest_trading_date = None
            closest_price = None
            
            # Convert news date to string for comparison
            news_date_str_formatted = news_date.strftime('%Y-%m-%d')
            
            # Find the closest date in our price data
            for j, (date, price) in enumerate(zip(dates, prices)):
                date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
                if date_str == news_date_str_formatted:
                    closest_trading_date = date
                    closest_price = price
                    break
            
            # If exact date not found, find the closest date within 1 day
            if closest_trading_date is None:
                news_date_obj = news_date.date()
                
                for j, (date, price) in enumerate(zip(dates, prices)):
                    date_obj = date.date() if hasattr(date, 'date') else datetime.strptime(str(date), '%Y-%m-%d').date()
                    date_diff = abs((date_obj - news_date_obj).days)
                    
                    if date_diff <= 1:  # Within 1 day
                        closest_trading_date = date
                        closest_price = price
                        break
            
            # Only add if we found a matching trading date
            if closest_trading_date is not None and closest_price is not None:
                date_key = closest_trading_date.strftime('%Y-%m-%d')
                
                # Store the latest news for each trading day
                if date_key not in real_news_by_date:
                    real_news_by_date[date_key] = {
                        'date': closest_trading_date,
                        'ticker': symbol,
                        'news': news_headline,
                        'close_price': closest_price,
                        'news_time': news_date,
                        'is_real': True
                    }
                else:
                    # If we already have news for this date, keep the latest one
                    if news_date > real_news_by_date[date_key]['news_time']:
                        real_news_by_date[date_key] = {
                            'date': closest_trading_date,
            'ticker': symbol,
            'news': news_headline,
                            'close_price': closest_price,
                            'news_time': news_date,
                            'is_real': True
                        }
                        
        except (KeyError, TypeError, ValueError) as e:
            print(f"⚠️ Error processing news item: {e}")
            continue
    
    real_news_count = len(real_news_by_date)
    print(f"📊 Real news matched: {real_news_count} trading days")
    
    # If we don't have enough real news, we need to supplement
    if real_news_count < min_days:
        print(f"⚠️ Only {real_news_count} days with real news, need {min_days}+ for meaningful backtesting")
        print(f"🔄 This ticker ({symbol}) doesn't have sufficient real news coverage")
        return []
    
    # Convert to list and sort by date
    news_data = []
    for date_key in sorted(real_news_by_date.keys()):
        item = real_news_by_date[date_key]
        news_data.append({
            'date': item['date'],
            'ticker': item['ticker'],
            'news': item['news'],
            'close_price': item['close_price']
        })
    
    print(f"✅ Successfully created dataset with {len(news_data)} trading days of REAL news for {symbol}")
    return news_data

def collect_financial_data():
    """Collect REAL financial data using tickers with sufficient news coverage"""
    print("🔍 Collecting REAL financial data using tickers with sufficient news coverage...")
    print("📝 Note: Only tickers with 100+ days of real news will be included")
    print("🆓 Using ONLY real sources: Hugging Face + Yahoo Finance (no synthetic generation)")
    print("💰 All prices are REAL market data from Yahoo Finance")
    
    # Find tickers with comprehensive news coverage
    symbols = find_top_tickers_in_huggingface()
    print(f"🎯 Analyzing tickers: {symbols}")
    
    # Set date range to cover the full Hugging Face dataset (2006-2024)
    start_date = "2006-01-01"  # Start from beginning of Hugging Face dataset
    end_date = "2024-12-31"    # End at the latest date in Hugging Face dataset
    
    print(f"📅 Date range: {start_date} to {end_date}")
    
    all_data = []
    successful_tickers = []
    
    for symbol in symbols:
        print(f"\n📈 Analyzing {symbol} for sufficient news coverage...")
        data = get_data(symbol, start_date, end_date)
        
        if data is not None and not data.empty:
            dates = data.index
            prices = data['Close'].values
            print(f"📊 Found {len(dates)} trading days for {symbol}")
            
            # Try to get comprehensive news data with minimum 100 days
            news_data = create_comprehensive_news_data_with_minimum(symbol, dates, prices, start_date, end_date, min_days=100)
            
            if len(news_data) >= 100:
                all_data.extend(news_data)
                successful_tickers.append(symbol)
                print(f"✅ {symbol}: {len(news_data)} trading days with REAL news (SUFFICIENT)")
            else:
                print(f"❌ {symbol}: Only {len(news_data)} days with real news (INSUFFICIENT)")
        else:
            print(f"❌ No price data found for {symbol}")
    
    if not all_data:
        print(f"\n⚠️ No tickers found with sufficient real news coverage (100+ days)")
        print(f"🔄 Falling back to any tickers with 50+ days of real news...")
        
        # Try with lower threshold
        for symbol in symbols:
            print(f"\n📈 Retrying {symbol} with lower threshold (50+ days)...")
            data = get_data(symbol, start_date, end_date)
            
            if data is not None and not data.empty:
                dates = data.index
                prices = data['Close'].values
                
                news_data = create_comprehensive_news_data_with_minimum(symbol, dates, prices, start_date, end_date, min_days=50)
                
                if len(news_data) >= 50:
                    all_data.extend(news_data)
                    successful_tickers.append(symbol)
                    print(f"✅ {symbol}: {len(news_data)} trading days with REAL news (ACCEPTABLE)")
                else:
                    print(f"❌ {symbol}: Only {len(news_data)} days with real news (INSUFFICIENT)")
    
    if not all_data:
        print(f"\n❌ No tickers found with sufficient real news coverage")
        print(f"📊 The Hugging Face dataset has limited coverage per ticker")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    output_file = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Probes/financial_trading_data.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"📁 Saved {len(df)} records to {output_file}")
    print(f"🎯 Successful tickers: {successful_tickers}")
    print(f"📰 ALL news headlines are REAL (from Hugging Face + Yahoo Finance)")
    print(f"💰 All close prices are REAL market data from Yahoo Finance")
    print(f"📅 Only trading days with REAL news are included")
    print(f"🕐 Each record represents the last REAL news of that trading day")
    print(f"🚫 NO synthetic or generated content included")
    
    return df

if __name__ == "__main__":
    collect_financial_data()
