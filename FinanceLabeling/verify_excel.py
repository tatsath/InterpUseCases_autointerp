#!/usr/bin/env python3
"""
Verify the structure of the combined Excel file
"""

import pandas as pd

def verify_excel():
    excel_file = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/FinanceLabeling/combined_analysis_results.xlsx"
    
    print("📊 Verifying Excel file structure...")
    
    # Read both sheets
    financial_df = pd.read_excel(excel_file, sheet_name='Financial_labels')
    general_df = pd.read_excel(excel_file, sheet_name='General_labels')
    
    print(f"\n📈 Financial_labels tab:")
    print(f"   Shape: {financial_df.shape}")
    print(f"   Columns: {list(financial_df.columns)}")
    print(f"   Sample data:")
    print(financial_df.head(3).to_string())
    
    print(f"\n📋 General_labels tab:")
    print(f"   Shape: {general_df.shape}")
    print(f"   Columns: {list(general_df.columns)}")
    print(f"   Sample data:")
    print(general_df.head(3).to_string())
    
    print(f"\n🎯 Summary:")
    print(f"   Financial features: {len(financial_df)}")
    print(f"   General features: {len(general_df)}")
    print(f"   Total features: {len(financial_df) + len(general_df)}")
    
    # Check sorting
    print(f"\n🔍 Sorting verification:")
    financial_sorted = financial_df.sort_values(['layer', 'feature'])
    general_sorted = general_df.sort_values(['layer', 'feature'])
    
    print(f"   Financial data is sorted: {financial_df.equals(financial_sorted)}")
    print(f"   General data is sorted: {general_df.equals(general_sorted)}")

if __name__ == "__main__":
    verify_excel()
