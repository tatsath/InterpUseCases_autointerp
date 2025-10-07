#!/usr/bin/env python3
"""
Simple runner for Finance Agents
"""

import os
from llama_finance_agents import SimpleFinanceAgents

def run_stock_analysis():
    """Run stock analysis example"""
    
    # Initialize agents
    agents = SimpleFinanceAgents()
    agents.setup()
    
    # Analyze stocks
    stocks_to_analyze = ["AAPL", "MSFT", "GOOGL"]
    
    for stock in stocks_to_analyze:
        print(f"\n{'='*60}")
        print(f"ANALYZING {stock}")
        print(f"{'='*60}")
        
        try:
            result = agents.analyze_stock(stock)
            print(result)
        except Exception as e:
            print(f"Error analyzing {stock}: {e}")
        
        print("\n" + "="*60)

def run_sector_analysis():
    """Run sector analysis example"""
    
    # Initialize agents
    agents = SimpleFinanceAgents()
    agents.setup()
    
    # Analyze sectors
    sectors_to_analyze = ["technology", "healthcare", "financial"]
    
    for sector in sectors_to_analyze:
        print(f"\n{'='*60}")
        print(f"ANALYZING {sector.upper()} SECTOR")
        print(f"{'='*60}")
        
        try:
            result = agents.analyze_sector(sector)
            print(result)
        except Exception as e:
            print(f"Error analyzing {sector}: {e}")
        
        print("\n" + "="*60)

if __name__ == "__main__":
    print("Simple Finance Agents System")
    print("Choose analysis type:")
    print("1. Stock Analysis")
    print("2. Sector Analysis")
    print("3. Both")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        run_stock_analysis()
    elif choice == "2":
        run_sector_analysis()
    elif choice == "3":
        run_stock_analysis()
        run_sector_analysis()
    else:
        print("Invalid choice. Running stock analysis...")
        run_stock_analysis()
