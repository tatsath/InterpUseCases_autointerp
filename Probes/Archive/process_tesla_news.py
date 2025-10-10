#!/usr/bin/env python3
"""
Process Tesla news articles from Excel file and combine with Yahoo Finance closing prices
"""

import pandas as pd
import yfinance as yf
from datetime import datetime
import os

def get_tesla_closing_prices(start_date, end_date):
    """Get Tesla closing prices from Yahoo Finance"""
    try:
        print(f"📈 Fetching TSLA closing prices from {start_date} to {end_date}...")
        
        # Download Tesla data
        tesla = yf.download("TSLA", start=start_date, end=end_date, progress=False)
        
        if tesla.empty:
            print("❌ No price data found for TSLA")
            return None
        
        print(f"📊 Found {len(tesla)} trading days of price data")
        return tesla
        
    except Exception as e:
        print(f"❌ Error fetching TSLA price data: {e}")
        return None

def process_tesla_news_excel():
    """Process Tesla news from Excel file and combine with closing prices"""
    print("🔍 Processing Tesla news articles from Excel file...")
    
    # Read the Excel file
    excel_file = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Probes/tesla_news_articles_2010_2020 3.xlsx"
    
    try:
        print(f"📖 Reading Excel file: {excel_file}")
        df = pd.read_excel(excel_file)
        
        print(f"📊 Excel file loaded: {len(df)} rows")
        print(f"📋 Columns: {list(df.columns)}")
        print(f"📋 Sample data:")
        print(df.head())
        
        # Determine date and news columns
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        news_columns = [col for col in df.columns if 'news' in col.lower() or 'headline' in col.lower() or 'title' in col.lower() or 'text' in col.lower()]
        
        print(f"📅 Date columns found: {date_columns}")
        print(f"📰 News columns found: {news_columns}")
        
        if not date_columns or not news_columns:
            print("❌ Could not identify date and news columns")
            return None
        
        # Use the first available date and news columns
        date_col = date_columns[0]
        news_col = news_columns[0]
        
        print(f"📅 Using date column: {date_col}")
        print(f"📰 Using news column: {news_col}")
        
        # Convert date column to datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Get date range
        start_date = df[date_col].min().strftime('%Y-%m-%d')
        end_date = df[date_col].max().strftime('%Y-%m-%d')
        
        print(f"📅 Date range: {start_date} to {end_date}")
        
        # Get Tesla closing prices
        price_data = get_tesla_closing_prices(start_date, end_date)
        
        if price_data is None:
            print("❌ Could not fetch price data")
            return None
        
        # Process each news article
        processed_data = []
        
        for idx, row in df.iterrows():
            try:
                news_date = row[date_col]
                news_headline = str(row[news_col])
                
                # Find the closest trading date
                closest_trading_date = None
                closest_price = None
                
                # Convert news date to string for comparison
                news_date_str = news_date.strftime('%Y-%m-%d')
                
                # Find exact date match first
                if news_date_str in price_data.index.strftime('%Y-%m-%d'):
                    closest_trading_date = news_date
                    closest_price = price_data.loc[news_date_str, 'Close']
                else:
                    # Find closest trading date within 3 days
                    news_date_obj = news_date.date()
                    
                    for trading_date in price_data.index:
                        trading_date_obj = trading_date.date()
                        date_diff = abs((trading_date_obj - news_date_obj).days)
                        
                        if date_diff <= 3:  # Within 3 days
                            closest_trading_date = trading_date
                            closest_price = price_data.loc[trading_date, 'Close']
                            break
                
                # Only add if we found a matching trading date
                if closest_trading_date is not None and closest_price is not None:
                    processed_data.append({
                        'date': closest_trading_date,
                        'ticker': 'TSLA',
                        'news': news_headline,
                        'close_price': float(closest_price)
                    })
                    
            except Exception as e:
                print(f"⚠️ Error processing row {idx}: {e}")
                continue
        
        print(f"📊 Successfully processed {len(processed_data)} news articles with matching prices")
        
        # Create DataFrame and save
        result_df = pd.DataFrame(processed_data)
        
        # Sort by date
        result_df = result_df.sort_values('date').reset_index(drop=True)
        
        # Save to CSV
        output_file = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Probes/Financial_data_Tesla.csv"
        result_df.to_csv(output_file, index=False)
        
        print(f"\n🎯 FINAL RESULTS:")
        print(f"📁 Saved {len(result_df)} records to {output_file}")
        print(f"📅 Date range: {result_df['date'].min()} to {result_df['date'].max()}")
        print(f"💰 Price range: ${result_df['close_price'].min():.2f} to ${result_df['close_price'].max():.2f}")
        print(f"📰 All news headlines are from the original Excel file")
        print(f"💰 All closing prices are from Yahoo Finance")
        
        # Show sample data
        print(f"\n📋 Sample data:")
        print(result_df.head(10))
        
        return result_df
        
    except Exception as e:
        print(f"❌ Error processing Excel file: {e}")
        return None

if __name__ == "__main__":
    process_tesla_news_excel()
