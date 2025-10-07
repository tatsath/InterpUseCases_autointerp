#!/usr/bin/env python3
"""
Simple Finance Agents System
Streamlined system for financial analysis using basic tools
"""

import os
import asyncio
import yfinance as yf
import pandas as pd
from typing import List, Dict, Any
import json

class SimpleFinanceAgents:
    """Simple finance agents system for financial analysis"""
    
    def __init__(self):
        self.agents = {}
        self.tools = {}
        
    def _create_tools(self):
        """Create financial analysis tools"""
        
        def get_stock_price(ticker: str) -> Dict[str, Any]:
            """Get current stock price and basic info"""
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                hist = stock.history(period="5d")
                
                return {
                    "ticker": ticker,
                    "current_price": info.get('currentPrice', 'N/A'),
                    "previous_close": info.get('previousClose', 'N/A'),
                    "market_cap": info.get('marketCap', 'N/A'),
                    "volume": info.get('volume', 'N/A'),
                    "52_week_high": info.get('fiftyTwoWeekHigh', 'N/A'),
                    "52_week_low": info.get('fiftyTwoWeekLow', 'N/A'),
                    "recent_prices": hist['Close'].tail(5).to_dict()
                }
            except Exception as e:
                return {"error": f"Failed to get data for {ticker}: {str(e)}"}
        
        def get_stock_news(ticker: str) -> List[Dict[str, Any]]:
            """Get recent news for a stock"""
            try:
                stock = yf.Ticker(ticker)
                news = stock.news
                return news[:5]  # Return top 5 news items
            except Exception as e:
                return [{"error": f"Failed to get news for {ticker}: {str(e)}"}]
        
        def get_financial_metrics(ticker: str) -> Dict[str, Any]:
            """Get key financial metrics"""
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                return {
                    "ticker": ticker,
                    "pe_ratio": info.get('trailingPE', 'N/A'),
                    "forward_pe": info.get('forwardPE', 'N/A'),
                    "peg_ratio": info.get('pegRatio', 'N/A'),
                    "price_to_book": info.get('priceToBook', 'N/A'),
                    "debt_to_equity": info.get('debtToEquity', 'N/A'),
                    "return_on_equity": info.get('returnOnEquity', 'N/A'),
                    "revenue_growth": info.get('revenueGrowth', 'N/A'),
                    "earnings_growth": info.get('earningsGrowth', 'N/A'),
                    "profit_margins": info.get('profitMargins', 'N/A')
                }
            except Exception as e:
                return {"error": f"Failed to get financial metrics for {ticker}: {str(e)}"}
        
        def analyze_sector_performance(sector: str) -> Dict[str, Any]:
            """Analyze sector performance"""
            try:
                # Get sector ETFs for analysis
                sector_etfs = {
                    "technology": "XLK",
                    "healthcare": "XLV", 
                    "financial": "XLF",
                    "energy": "XLE",
                    "consumer": "XLY",
                    "industrial": "XLI"
                }
                
                if sector.lower() not in sector_etfs:
                    return {"error": f"Sector {sector} not supported"}
                
                etf_ticker = sector_etfs[sector.lower()]
                etf = yf.Ticker(etf_ticker)
                hist = etf.history(period="1mo")
                
                return {
                    "sector": sector,
                    "etf_ticker": etf_ticker,
                    "monthly_return": ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100,
                    "volatility": hist['Close'].pct_change().std() * 100,
                    "current_price": hist['Close'].iloc[-1]
                }
            except Exception as e:
                return {"error": f"Failed to analyze sector {sector}: {str(e)}"}
        
        # Store tools
        self.tools = {
            "get_stock_price": get_stock_price,
            "get_stock_news": get_stock_news,
            "get_financial_metrics": get_financial_metrics,
            "analyze_sector_performance": analyze_sector_performance
        }
    
    def _create_agents(self):
        """Create specialized finance agents"""
        
        # Market Data Agent
        self.agents['market_data'] = {
            "name": "Market_Data_Agent",
            "description": "Specialized in gathering stock prices, financial metrics, and market data",
            "tools": ["get_stock_price", "get_financial_metrics"],
            "system_message": "You are a financial data specialist. Use your tools to gather accurate stock prices, financial metrics, and market information. Provide clear, factual data without speculation."
        }
        
        # News Analysis Agent  
        self.agents['news_analysis'] = {
            "name": "News_Analysis_Agent", 
            "description": "Specialized in gathering and analyzing financial news",
            "tools": ["get_stock_news"],
            "system_message": "You are a financial news analyst. Use your tools to gather recent news and provide analysis of market sentiment and potential impacts."
        }
        
        # Sector Analysis Agent
        self.agents['sector_analysis'] = {
            "name": "Sector_Analysis_Agent",
            "description": "Specialized in sector performance analysis and market trends",
            "tools": ["analyze_sector_performance"],
            "system_message": "You are a sector analyst. Use your tools to analyze sector performance, identify trends, and provide insights on market movements."
        }
        
        # Investment Advisor Agent
        self.agents['investment_advisor'] = {
            "name": "Investment_Advisor_Agent",
            "description": "Provides investment recommendations and portfolio analysis",
            "tools": ["get_stock_price", "get_financial_metrics", "get_stock_news", "analyze_sector_performance"],
            "system_message": "You are an investment advisor. Analyze the data provided by other agents to give investment recommendations, risk assessments, and portfolio insights. Always include disclaimers about investment risks."
        }
    
    def setup(self):
        """Setup the complete agent system"""
        print("Setting up Simple Finance Agents...")
        self._create_tools()
        self._create_agents()
        print("✅ Finance agents system ready!")
    
    def analyze_stock(self, ticker: str) -> str:
        """Analyze a stock using the agent system"""
        print(f"Starting analysis for {ticker}...")
        
        # Gather data using tools
        stock_data = self.tools["get_stock_price"](ticker)
        financial_metrics = self.tools["get_financial_metrics"](ticker)
        news_data = self.tools["get_stock_news"](ticker)
        
        # Create analysis report
        report = []
        report.append("="*80)
        report.append(f"FINANCIAL ANALYSIS REPORT FOR {ticker.upper()}")
        report.append("="*80)
        
        # Market Data Analysis
        report.append("\n📊 MARKET DATA ANALYSIS")
        report.append("-" * 40)
        if "error" not in stock_data:
            report.append(f"Current Price: ${stock_data.get('current_price', 'N/A')}")
            report.append(f"Previous Close: ${stock_data.get('previous_close', 'N/A')}")
            report.append(f"Market Cap: ${stock_data.get('market_cap', 'N/A'):,}" if isinstance(stock_data.get('market_cap'), (int, float)) else f"Market Cap: {stock_data.get('market_cap', 'N/A')}")
            report.append(f"Volume: {stock_data.get('volume', 'N/A'):,}" if isinstance(stock_data.get('volume'), (int, float)) else f"Volume: {stock_data.get('volume', 'N/A')}")
            report.append(f"52-Week High: ${stock_data.get('52_week_high', 'N/A')}")
            report.append(f"52-Week Low: ${stock_data.get('52_week_low', 'N/A')}")
        else:
            report.append(f"Error: {stock_data['error']}")
        
        # Financial Metrics Analysis
        report.append("\n📈 FINANCIAL METRICS")
        report.append("-" * 40)
        if "error" not in financial_metrics:
            report.append(f"P/E Ratio: {financial_metrics.get('pe_ratio', 'N/A')}")
            report.append(f"Forward P/E: {financial_metrics.get('forward_pe', 'N/A')}")
            report.append(f"PEG Ratio: {financial_metrics.get('peg_ratio', 'N/A')}")
            report.append(f"Price-to-Book: {financial_metrics.get('price_to_book', 'N/A')}")
            report.append(f"Debt-to-Equity: {financial_metrics.get('debt_to_equity', 'N/A')}")
            report.append(f"Return on Equity: {financial_metrics.get('return_on_equity', 'N/A')}")
            report.append(f"Revenue Growth: {financial_metrics.get('revenue_growth', 'N/A')}")
            report.append(f"Earnings Growth: {financial_metrics.get('earnings_growth', 'N/A')}")
            report.append(f"Profit Margins: {financial_metrics.get('profit_margins', 'N/A')}")
        else:
            report.append(f"Error: {financial_metrics['error']}")
        
        # News Analysis
        report.append("\n📰 RECENT NEWS ANALYSIS")
        report.append("-" * 40)
        if "error" not in news_data[0] if news_data else True:
            for i, news_item in enumerate(news_data[:3], 1):  # Show top 3 news items
                report.append(f"{i}. {news_item.get('title', 'No title')}")
                report.append(f"   Published: {news_item.get('providerPublishTime', 'Unknown date')}")
                report.append(f"   Source: {news_item.get('publisher', 'Unknown source')}")
                report.append("")
        else:
            report.append(f"Error: {news_data[0]['error'] if news_data else 'No news available'}")
        
        # Investment Recommendation
        report.append("\n💡 INVESTMENT RECOMMENDATION")
        report.append("-" * 40)
        report.append("Based on the analysis above, here are key insights:")
        
        if "error" not in stock_data and "error" not in financial_metrics:
            current_price = stock_data.get('current_price', 0)
            pe_ratio = financial_metrics.get('pe_ratio', 0)
            
            if isinstance(pe_ratio, (int, float)) and pe_ratio > 0:
                if pe_ratio < 15:
                    report.append("• P/E ratio suggests the stock may be undervalued")
                elif pe_ratio > 25:
                    report.append("• P/E ratio suggests the stock may be overvalued")
                else:
                    report.append("• P/E ratio appears reasonable")
            
            report.append("• Consider the company's financial health and growth prospects")
            report.append("• Review recent news for any significant developments")
            report.append("• Assess sector trends and market conditions")
        
        report.append("\n⚠️  DISCLAIMER: This analysis is for informational purposes only.")
        report.append("   Not financial advice. Always do your own research and consult")
        report.append("   with a qualified financial advisor before making investment decisions.")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)
    
    def analyze_sector(self, sector: str) -> str:
        """Analyze a sector using the agent system"""
        print(f"Starting sector analysis for {sector}...")
        
        # Gather sector data
        sector_data = self.tools["analyze_sector_performance"](sector)
        
        # Create sector analysis report
        report = []
        report.append("="*80)
        report.append(f"SECTOR ANALYSIS REPORT FOR {sector.upper()}")
        report.append("="*80)
        
        if "error" not in sector_data:
            report.append(f"\n📊 SECTOR PERFORMANCE METRICS")
            report.append("-" * 40)
            report.append(f"Sector: {sector_data.get('sector', 'N/A')}")
            report.append(f"ETF Ticker: {sector_data.get('etf_ticker', 'N/A')}")
            report.append(f"Monthly Return: {sector_data.get('monthly_return', 'N/A'):.2f}%")
            report.append(f"Volatility: {sector_data.get('volatility', 'N/A'):.2f}%")
            report.append(f"Current ETF Price: ${sector_data.get('current_price', 'N/A'):.2f}")
            
            # Analysis insights
            monthly_return = sector_data.get('monthly_return', 0)
            volatility = sector_data.get('volatility', 0)
            
            report.append(f"\n💡 SECTOR INSIGHTS")
            report.append("-" * 40)
            if monthly_return > 5:
                report.append("• Strong positive performance this month")
            elif monthly_return < -5:
                report.append("• Significant decline this month")
            else:
                report.append("• Moderate performance this month")
            
            if volatility > 20:
                report.append("• High volatility - consider risk management")
            elif volatility < 10:
                report.append("• Low volatility - relatively stable sector")
            else:
                report.append("• Moderate volatility")
            
            report.append(f"\n📈 INVESTMENT CONSIDERATIONS")
            report.append("-" * 40)
            report.append("• Review individual stocks within this sector")
            report.append("• Consider sector-specific risks and opportunities")
            report.append("• Monitor economic indicators affecting this sector")
            report.append("• Evaluate diversification across sectors")
            
        else:
            report.append(f"Error: {sector_data['error']}")
        
        report.append("\n⚠️  DISCLAIMER: This analysis is for informational purposes only.")
        report.append("   Not financial advice. Always do your own research and consult")
        report.append("   with a qualified financial advisor before making investment decisions.")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)

def main():
    """Main function to run the finance agents"""
    # Initialize the system
    finance_agents = SimpleFinanceAgents()
    finance_agents.setup()
    
    # Example usage
    print("\n" + "="*80)
    print("SIMPLE FINANCE AGENTS SYSTEM")
    print("="*80)
    
    # Analyze a stock
    ticker = "AAPL"  # Apple
    print(f"\nAnalyzing {ticker}...")
    result = finance_agents.analyze_stock(ticker)
    print(result)
    
    # Analyze a sector
    sector = "technology"
    print(f"\nAnalyzing {sector} sector...")
    result = finance_agents.analyze_sector(sector)
    print(result)

if __name__ == "__main__":
    main()