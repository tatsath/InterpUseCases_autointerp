#!/usr/bin/env python3
"""
Create a combined Excel file with two tabs:
1. Financial_labels: Results from Yahoo Finance analysis (sorted by layer, then feature)
2. General_labels: Results from general analysis for each layer (F1 scores only)
"""

import pandas as pd
import os
from pathlib import Path

def create_combined_excel():
    # File paths
    financial_csv = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/FinanceLabeling/all_layers_financial_results/all_layers_all_features_large_model_yahoo_finance/results_summary.csv"
    general_csvs = {
        4: "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/FinanceLabeling/multi_layer_full_results/results_summary_layer4.csv",
        10: "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/FinanceLabeling/multi_layer_full_results/results_summary_layer10.csv",
        16: "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/FinanceLabeling/multi_layer_full_results/results_summary_layer16.csv",
        22: "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/FinanceLabeling/multi_layer_full_results/results_summary_layer22.csv",
        28: "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/FinanceLabeling/multi_layer_full_results/results_summary_layer28.csv"
    }
    
    output_file = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/FinanceLabeling/combined_analysis_results.xlsx"
    
    print("📊 Creating combined Excel file...")
    
    # Read financial labels data
    print("📈 Reading financial labels data...")
    financial_df = pd.read_csv(financial_csv)
    
    # Sort financial data by layer, then by feature
    financial_df_sorted = financial_df.sort_values(['layer', 'feature']).reset_index(drop=True)
    print(f"✅ Financial data: {len(financial_df_sorted)} rows")
    
    # Read and combine general labels data
    print("📋 Reading general labels data...")
    general_dfs = []
    
    for layer, csv_path in general_csvs.items():
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Keep only F1 score as requested
            df_f1_only = df[['layer', 'feature', 'label', 'f1_score']].copy()
            general_dfs.append(df_f1_only)
            print(f"✅ Layer {layer}: {len(df_f1_only)} features")
        else:
            print(f"⚠️  Warning: File not found for layer {layer}: {csv_path}")
    
    # Combine all general data
    if general_dfs:
        general_combined = pd.concat(general_dfs, ignore_index=True)
        # Sort by layer, then by feature
        general_combined_sorted = general_combined.sort_values(['layer', 'feature']).reset_index(drop=True)
        print(f"✅ General data combined: {len(general_combined_sorted)} rows")
    else:
        print("❌ No general data found!")
        general_combined_sorted = pd.DataFrame()
    
    # Create Excel file with two tabs
    print("📝 Writing Excel file...")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Financial labels tab
        financial_df_sorted.to_excel(writer, sheet_name='Financial_labels', index=False)
        print("✅ Financial_labels tab created")
        
        # General labels tab (F1 scores only)
        if not general_combined_sorted.empty:
            general_combined_sorted.to_excel(writer, sheet_name='General_labels', index=False)
            print("✅ General_labels tab created")
        else:
            # Create empty sheet if no data
            pd.DataFrame({'message': ['No general data available']}).to_excel(writer, sheet_name='General_labels', index=False)
            print("⚠️  General_labels tab created (empty)")
    
    print(f"🎉 Combined Excel file created: {output_file}")
    
    # Print summary
    print("\n📊 Summary:")
    print(f"   Financial_labels: {len(financial_df_sorted)} features")
    print(f"   General_labels: {len(general_combined_sorted)} features")
    print(f"   Total features: {len(financial_df_sorted) + len(general_combined_sorted)}")
    
    return output_file

if __name__ == "__main__":
    create_combined_excel()
