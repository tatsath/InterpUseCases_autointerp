#!/usr/bin/env python3
"""
Check Model Structure and Extract Feature Importance
"""

import joblib
import pandas as pd
import numpy as np

def check_model_structure():
    """Check what's in the saved model file"""
    
    try:
        # Load the model
        model_data = joblib.load('results/trading_model.pkl')
        
        print("="*80)
        print("MODEL STRUCTURE ANALYSIS")
        print("="*80)
        
        print(f"Model type: {type(model_data)}")
        print(f"Model keys: {list(model_data.keys()) if isinstance(model_data, dict) else 'Not a dict'}")
        
        if isinstance(model_data, dict):
            for key, value in model_data.items():
                print(f"\n{key}: {type(value)}")
                if hasattr(value, 'shape'):
                    print(f"  Shape: {value.shape}")
                elif isinstance(value, dict):
                    print(f"  Keys: {list(value.keys())}")
                elif isinstance(value, list):
                    print(f"  Length: {len(value)}")
        
        # Check if feature importance is stored
        if 'feature_importance' in model_data:
            print(f"\n🔍 FEATURE IMPORTANCE FOUND!")
            feature_importance = model_data['feature_importance']
            print(f"Feature importance type: {type(feature_importance)}")
            
            if isinstance(feature_importance, dict):
                for model_name, importance_dict in feature_importance.items():
                    print(f"\n{model_name}:")
                    if isinstance(importance_dict, dict):
                        # Convert to DataFrame and sort
                        importance_df = pd.DataFrame([
                            {'feature': feature, 'importance': importance}
                            for feature, importance in importance_dict.items()
                        ]).sort_values('importance', ascending=False)
                        
                        # Filter for SAE features
                        sae_features = importance_df[importance_df['feature'].str.contains('SAE_', na=False)]
                        
                        if not sae_features.empty:
                            print(f"  Top 10 SAE Features:")
                            print(sae_features.head(10).to_string(index=False))
                        else:
                            print("  No SAE features found")
                    else:
                        print(f"  Importance data type: {type(importance_dict)}")
        
        # Check if models are stored
        if 'models' in model_data:
            print(f"\n🤖 MODELS FOUND!")
            models = model_data['models']
            print(f"Models type: {type(models)}")
            
            if isinstance(models, dict):
                for model_name, model in models.items():
                    print(f"\n{model_name}: {type(model)}")
                    
                    # Try to get feature importance directly from the model
                    if hasattr(model, 'feature_importances_'):
                        print(f"  Has feature_importances_: {model.feature_importances_.shape}")
                        # Get feature names if available
                        if 'feature_names' in model_data:
                            feature_names = model_data['feature_names']
                            importance_df = pd.DataFrame({
                                'feature': feature_names,
                                'importance': model.feature_importances_
                            }).sort_values('importance', ascending=False)
                            
                            sae_features = importance_df[importance_df['feature'].str.contains('SAE_', na=False)]
                            if not sae_features.empty:
                                print(f"  Top 5 SAE Features:")
                                print(sae_features.head(5).to_string(index=False))
                    
                    elif hasattr(model, 'coef_'):
                        print(f"  Has coef_: {model.coef_.shape}")
                        if 'feature_names' in model_data:
                            feature_names = model_data['feature_names']
                            importance_df = pd.DataFrame({
                                'feature': feature_names,
                                'importance': np.abs(model.coef_[0])
                            }).sort_values('importance', ascending=False)
                            
                            sae_features = importance_df[importance_df['feature'].str.contains('SAE_', na=False)]
                            if not sae_features.empty:
                                print(f"  Top 5 SAE Features:")
                                print(sae_features.head(5).to_string(index=False))
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_model_structure()

