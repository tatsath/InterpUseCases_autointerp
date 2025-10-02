#!/usr/bin/env python3
"""
Setup Hallucination Detection for LLMProbe
Creates datasets for training linear probes to detect hallucinations
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def create_hallucination_dataset():
    """Create a dataset with factual vs hallucinated examples for probe training"""
    
    # Factual examples (label: 0)
    factual_examples = [
        "Apple Inc. was founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne.",
        "The Dow Jones Industrial Average is a stock market index of 30 large companies.",
        "Bitcoin was created in 2009 by an anonymous person or group using the name Satoshi Nakamoto.",
        "The Federal Reserve is the central bank of the United States.",
        "Tesla was founded in 2003 by Elon Musk and others.",
        "The S&P 500 is a stock market index that measures the performance of 500 large companies.",
        "Gold is a precious metal that has been used as currency for thousands of years.",
        "The NASDAQ is an American stock exchange based in New York City.",
        "Warren Buffett is known as the 'Oracle of Omaha' for his investment success.",
        "The SEC (Securities and Exchange Commission) regulates the securities industry."
    ]
    
    # Hallucinated examples (label: 1) - these contain false information
    hallucinated_examples = [
        "Apple Inc. was founded in 1985 by Bill Gates and Steve Jobs.",
        "The Dow Jones Industrial Average is a cryptocurrency index tracking 100 digital currencies.",
        "Bitcoin was created in 2015 by Elon Musk as a joke.",
        "The Federal Reserve is a private company owned by JPMorgan Chase.",
        "Tesla was founded in 1995 by Jeff Bezos and others.",
        "The S&P 500 is a bond market index that measures government debt performance.",
        "Gold is a synthetic metal created in laboratories in the 1950s.",
        "The NASDAQ is a European stock exchange based in London.",
        "Warren Buffett is known as the 'Wizard of Wall Street' for his trading algorithms.",
        "The SEC (Securities and Exchange Commission) is a private hedge fund."
    ]
    
    # Create dataset
    data = []
    
    # Add factual examples
    for example in factual_examples:
        data.append({
            'text': example,
            'label': 0,  # 0 = factual
            'type': 'factual'
        })
    
    # Add hallucinated examples  
    for example in hallucinated_examples:
        data.append({
            'text': example,
            'label': 1,  # 1 = hallucinated
            'type': 'hallucinated'
        })
    
    return pd.DataFrame(data)

def create_financial_hallucination_dataset():
    """Create financial-specific hallucination examples"""
    
    # Financial factual examples
    financial_factual = [
        "The Federal Reserve sets interest rates to control inflation.",
        "Stock prices are determined by supply and demand in the market.",
        "Diversification reduces investment risk by spreading it across assets.",
        "Compound interest allows investments to grow exponentially over time.",
        "The SEC requires public companies to file quarterly earnings reports.",
        "Bonds are debt securities that pay fixed interest over time.",
        "Market capitalization is calculated by multiplying shares outstanding by stock price.",
        "The P/E ratio compares a company's stock price to its earnings per share.",
        "Inflation reduces the purchasing power of money over time.",
        "The S&P 500 is a market-capitalization-weighted index."
    ]
    
    # Financial hallucinated examples
    financial_hallucinated = [
        "The Federal Reserve sets interest rates to increase unemployment.",
        "Stock prices are determined by the weather forecast.",
        "Diversification increases investment risk by concentrating holdings.",
        "Compound interest causes investments to shrink over time.",
        "The SEC requires companies to file daily earnings reports.",
        "Bonds are equity securities that pay variable dividends.",
        "Market capitalization is calculated by adding all stock prices together.",
        "The P/E ratio compares a company's revenue to its stock price.",
        "Inflation increases the purchasing power of money over time.",
        "The S&P 500 is an equal-weighted index of all stocks."
    ]
    
    data = []
    
    # Add factual examples
    for example in financial_factual:
        data.append({
            'text': example,
            'label': 0,
            'type': 'financial_factual'
        })
    
    # Add hallucinated examples
    for example in financial_hallucinated:
        data.append({
            'text': example,
            'label': 1,
            'type': 'financial_hallucinated'
        })
    
    return pd.DataFrame(data)

def create_earnings_hallucination_dataset():
    """Create earnings report hallucination examples"""
    
    # Real earnings patterns
    earnings_factual = [
        "Apple reported Q3 revenue of $81.4 billion, up 1% year-over-year.",
        "Tesla's net income increased 20% compared to the previous quarter.",
        "Microsoft's cloud revenue grew 16% in the latest quarter.",
        "Amazon's AWS segment generated $22.1 billion in revenue.",
        "Google's advertising revenue declined 2% year-over-year.",
        "Meta's Reality Labs division reported operating losses of $3.7 billion.",
        "Netflix added 5.9 million subscribers in Q2 2024.",
        "Nvidia's data center revenue increased 171% year-over-year.",
        "Salesforce reported subscription revenue of $8.6 billion.",
        "Oracle's cloud infrastructure revenue grew 66% year-over-year."
    ]
    
    # Hallucinated earnings (impossible/improbable numbers)
    earnings_hallucinated = [
        "Apple reported Q3 revenue of $8.14 trillion, up 1000% year-over-year.",
        "Tesla's net income increased 2000% compared to the previous quarter.",
        "Microsoft's cloud revenue grew 1600% in the latest quarter.",
        "Amazon's AWS segment generated $2.21 trillion in revenue.",
        "Google's advertising revenue declined 200% year-over-year.",
        "Meta's Reality Labs division reported operating profits of $37 billion.",
        "Netflix added 590 million subscribers in Q2 2024.",
        "Nvidia's data center revenue increased 17100% year-over-year.",
        "Salesforce reported subscription revenue of $860 billion.",
        "Oracle's cloud infrastructure revenue grew 6600% year-over-year."
    ]
    
    data = []
    
    # Add factual examples
    for example in earnings_factual:
        data.append({
            'text': example,
            'label': 0,
            'type': 'earnings_factual'
        })
    
    # Add hallucinated examples
    for example in earnings_hallucinated:
        data.append({
            'text': example,
            'label': 1,
            'type': 'earnings_hallucinated'
        })
    
    return pd.DataFrame(data)

def main():
    """Create all hallucination detection datasets"""
    
    print("🔍 Creating Hallucination Detection Datasets for LLMProbe")
    print("=" * 60)
    
    # Create datasets
    general_df = create_hallucination_dataset()
    financial_df = create_financial_hallucination_dataset()
    earnings_df = create_earnings_hallucination_dataset()
    
    # Combine all datasets
    combined_df = pd.concat([general_df, financial_df, earnings_df], ignore_index=True)
    
    # Create output directory
    output_dir = Path("hallucination_datasets")
    output_dir.mkdir(exist_ok=True)
    
    # Save datasets
    general_df.to_csv(output_dir / "general_hallucination.csv", index=False)
    financial_df.to_csv(output_dir / "financial_hallucination.csv", index=False)
    earnings_df.to_csv(output_dir / "earnings_hallucination.csv", index=False)
    combined_df.to_csv(output_dir / "combined_hallucination.csv", index=False)
    
    # Create metadata for LLMProbe
    metadata = {
        "dataset_name": "hallucination_detection",
        "description": "Dataset for training linear probes to detect hallucinations",
        "total_examples": len(combined_df),
        "factual_examples": len(combined_df[combined_df['label'] == 0]),
        "hallucinated_examples": len(combined_df[combined_df['label'] == 1]),
        "features": {
            "text": "Input text to analyze",
            "label": "Binary label: 0=factual, 1=hallucinated",
            "type": "Category of example"
        },
        "usage": "Train linear probes on different model layers to detect hallucination patterns"
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Created {len(combined_df)} examples across {len(combined_df['type'].unique())} categories")
    print(f"📁 Saved datasets in: {output_dir}")
    print(f"📊 Factual examples: {len(combined_df[combined_df['label'] == 0])}")
    print(f"📊 Hallucinated examples: {len(combined_df[combined_df['label'] == 1])}")
    
    print("\n🎯 Next Steps:")
    print("1. Go to http://localhost:8501 (LLMProbe UI)")
    print("2. Upload the CSV files from hallucination_datasets/")
    print("3. Train linear probes on different layers")
    print("4. Analyze which layers best detect hallucinations")
    print("5. Use the probe weights to detect hallucinations in real-time")

if __name__ == "__main__":
    main()

