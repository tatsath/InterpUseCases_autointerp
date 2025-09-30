#!/usr/bin/env python3
"""
Extract Feature Importance from Trained Model
Shows which SAE features were most important for the actual prediction model
"""

import pandas as pd
import numpy as np
import joblib
import json
from prediction_model import TradingPredictionModel

def extract_feature_importance():
    """Extract and analyze feature importance from the trained model"""
    
    try:
        # Load the trained model
        model = joblib.load('results/trading_model.pkl')
        
        print("="*80)
        print("SAE FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        # Get feature importance for each model
        for model_name in ['logistic', 'random_forest', 'xgboost', 'gradient_boosting']:
            print(f"\n🔍 {model_name.upper()} MODEL FEATURE IMPORTANCE:")
            print("-" * 50)
            
            try:
                importance_df = model.get_feature_importance(model_name)
                if importance_df is not None and not importance_df.empty:
                    # Filter for SAE features only
                    sae_features = importance_df[importance_df['feature'].str.contains('SAE_', na=False)]
                    
                    if not sae_features.empty:
                        print(f"Top 10 SAE Features for {model_name}:")
                        print(sae_features.head(10).to_string(index=False))
                        
                        # Get top SAE feature
                        top_sae = sae_features.iloc[0]
                        print(f"\n🏆 Most Important SAE Feature: {top_sae['feature']} (Importance: {top_sae['importance']:.4f})")
                    else:
                        print("No SAE features found in importance data")
                else:
                    print("No feature importance data available")
            except Exception as e:
                print(f"Error getting importance for {model_name}: {e}")
        
        # Get overall feature importance across all models
        print(f"\n🌐 OVERALL FEATURE IMPORTANCE (All Models Combined):")
        print("-" * 60)
        
        try:
            all_importance = model.get_feature_importance()
            if all_importance is not None and not all_importance.empty:
                # Filter for SAE features
                sae_features_all = all_importance[all_importance['feature'].str.contains('SAE_', na=False)]
                
                if not sae_features_all.empty:
                    # Group by feature and calculate average importance
                    sae_avg = sae_features_all.groupby('feature')['importance'].agg(['mean', 'std', 'count']).reset_index()
                    sae_avg = sae_avg.sort_values('mean', ascending=False)
                    
                    print("Top 15 SAE Features (Average Importance across all models):")
                    print(sae_avg.head(15).to_string(index=False))
                    
                    # Save to file
                    sae_avg.to_csv('results/sae_feature_importance.csv', index=False)
                    print(f"\n💾 Results saved to: results/sae_feature_importance.csv")
                    
                    # Show top 5 with details
                    print(f"\n🏆 TOP 5 MOST IMPORTANT SAE FEATURES:")
                    print("=" * 50)
                    for i, row in sae_avg.head(5).iterrows():
                        feature_name = row['feature']
                        avg_importance = row['mean']
                        std_importance = row['std']
                        model_count = row['count']
                        
                        print(f"{i+1}. {feature_name}")
                        print(f"   Average Importance: {avg_importance:.4f}")
                        print(f"   Std Deviation: {std_importance:.4f}")
                        print(f"   Used in {model_count} models")
                        print()
                else:
                    print("No SAE features found in overall importance data")
            else:
                print("No overall feature importance data available")
        except Exception as e:
            print(f"Error getting overall importance: {e}")
        
        # Also show top technical indicators for comparison
        print(f"\n📊 TOP TECHNICAL INDICATORS (for comparison):")
        print("-" * 50)
        
        try:
            if all_importance is not None and not all_importance.empty:
                # Filter for non-SAE features (technical indicators)
                tech_features = all_importance[~all_importance['feature'].str.contains('SAE_', na=False)]
                
                if not tech_features.empty:
                    tech_avg = tech_features.groupby('feature')['importance'].agg(['mean', 'std', 'count']).reset_index()
                    tech_avg = tech_avg.sort_values('mean', ascending=False)
                    
                    print("Top 10 Technical Indicators:")
                    print(tech_avg.head(10).to_string(index=False))
        except Exception as e:
            print(f"Error getting technical indicators: {e}")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the model file exists at results/trading_model.pkl")

if __name__ == "__main__":
    extract_feature_importance()

