#!/usr/bin/env python3
"""
Analyze SAE Feature Importance from Trained Models
"""

import joblib
import pandas as pd
import numpy as np

def analyze_sae_importance():
    """Analyze which SAE features were most important for the actual prediction model"""
    
    try:
        # Load the model
        model_data = joblib.load('results/trading_model.pkl')
        feature_importance = model_data['feature_importance']
        
        print("="*80)
        print("🏆 MOST IMPORTANT SAE FEATURES FOR PREDICTION MODEL")
        print("="*80)
        
        # Collect all SAE features from all models
        all_sae_features = {}
        
        for model_name, importance_dict in feature_importance.items():
            print(f"\n📊 {model_name.upper()} MODEL:")
            print("-" * 50)
            
            # Convert to DataFrame and filter SAE features
            importance_df = pd.DataFrame([
                {'feature': feature, 'importance': importance}
                for feature, importance in importance_dict.items()
            ]).sort_values('importance', ascending=False)
            
            sae_features = importance_df[importance_df['feature'].str.contains('SAE_', na=False)]
            
            if not sae_features.empty:
                print("Top 10 SAE Features:")
                print(sae_features.head(10).to_string(index=False))
                
                # Store for overall analysis
                for _, row in sae_features.iterrows():
                    feature_name = row['feature']
                    importance = row['importance']
                    
                    if feature_name not in all_sae_features:
                        all_sae_features[feature_name] = []
                    all_sae_features[feature_name].append(importance)
            else:
                print("No SAE features found")
        
        # Calculate overall importance across all models
        print(f"\n🌐 OVERALL SAE FEATURE IMPORTANCE (Average across all models):")
        print("=" * 70)
        
        overall_importance = []
        for feature_name, importances in all_sae_features.items():
            avg_importance = np.mean(importances)
            std_importance = np.std(importances)
            model_count = len(importances)
            
            overall_importance.append({
                'feature': feature_name,
                'avg_importance': avg_importance,
                'std_importance': std_importance,
                'model_count': model_count,
                'max_importance': np.max(importances),
                'min_importance': np.min(importances)
            })
        
        # Sort by average importance
        overall_df = pd.DataFrame(overall_importance).sort_values('avg_importance', ascending=False)
        
        print("Top 20 Most Important SAE Features:")
        print(overall_df.head(20).to_string(index=False, float_format='%.6f'))
        
        # Save results
        overall_df.to_csv('results/sae_feature_importance_analysis.csv', index=False)
        print(f"\n💾 Results saved to: results/sae_feature_importance_analysis.csv")
        
        # Analyze by feature type
        print(f"\n🔍 ANALYSIS BY SAE FEATURE TYPE:")
        print("=" * 50)
        
        # Group by feature categories
        feature_categories = {
            'Earnings': overall_df[overall_df['feature'].str.contains('earnings', case=False)],
            'Private Equity': overall_df[overall_df['feature'].str.contains('private_equity', case=False)],
            'Revenue': overall_df[overall_df['feature'].str.contains('revenue', case=False)],
            'Cryptocurrency': overall_df[overall_df['feature'].str.contains('cryptocurrency', case=False)],
            'Foreign Exchange': overall_df[overall_df['feature'].str.contains('foreign_exchange', case=False)],
            'Inflation': overall_df[overall_df['feature'].str.contains('inflation', case=False)],
            'Composite': overall_df[overall_df['feature'].str.contains('composite', case=False)],
            'Layer 4': overall_df[overall_df['feature'].str.contains('layer_4', case=False)],
            'Layer 10': overall_df[overall_df['feature'].str.contains('layer_10', case=False)],
            'Layer 16': overall_df[overall_df['feature'].str.contains('layer_16', case=False)],
            'Layer 22': overall_df[overall_df['feature'].str.contains('layer_22', case=False)],
            'Layer 28': overall_df[overall_df['feature'].str.contains('layer_28', case=False)]
        }
        
        for category, df in feature_categories.items():
            if not df.empty:
                print(f"\n{category}:")
                print(f"  Count: {len(df)}")
                print(f"  Avg Importance: {df['avg_importance'].mean():.6f}")
                print(f"  Top Feature: {df.iloc[0]['feature']} ({df.iloc[0]['avg_importance']:.6f})")
        
        # Show top 5 most important features with details
        print(f"\n🏆 TOP 5 MOST IMPORTANT SAE FEATURES FOR PREDICTION:")
        print("=" * 60)
        
        for i, row in overall_df.head(5).iterrows():
            feature_name = row['feature']
            avg_importance = row['avg_importance']
            std_importance = row['std_importance']
            model_count = row['model_count']
            max_importance = row['max_importance']
            
            print(f"\n{i+1}. {feature_name}")
            print(f"   Average Importance: {avg_importance:.6f}")
            print(f"   Std Deviation: {std_importance:.6f}")
            print(f"   Used in {model_count}/4 models")
            print(f"   Max Importance: {max_importance:.6f}")
            
            # Interpret the feature
            if 'earnings' in feature_name.lower():
                print(f"   📈 Captures: Earnings-related market patterns")
            elif 'private_equity' in feature_name.lower():
                print(f"   💼 Captures: Private equity market dynamics")
            elif 'revenue' in feature_name.lower():
                print(f"   💰 Captures: Revenue and financial performance")
            elif 'cryptocurrency' in feature_name.lower():
                print(f"   🪙 Captures: Crypto-specific market behavior")
            elif 'foreign_exchange' in feature_name.lower():
                print(f"   🌍 Captures: Forex and international market impacts")
            elif 'inflation' in feature_name.lower():
                print(f"   📊 Captures: Inflation and economic indicators")
            elif 'composite' in feature_name.lower():
                print(f"   🔗 Captures: Combined multi-feature signals")
            
            if 'x_' in feature_name:
                print(f"   🔄 Interaction Feature: Combines SAE with technical indicators")
        
        return overall_df
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    analyze_sae_importance()

