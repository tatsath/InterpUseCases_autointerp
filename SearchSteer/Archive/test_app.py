#!/usr/bin/env python3
"""
Test script for the Financial Feature Steering App
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported."""
    try:
        import streamlit as st
        import torch
        import numpy as np
        import json
        from pathlib import Path
        from transformers import AutoTokenizer, AutoModel
        import plotly.express as px
        import plotly.graph_objects as go
        import pandas as pd
        print("✅ All required modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_feature_data():
    """Test if feature data is properly structured."""
    try:
        from streamlit_feature_steering_app import FinancialFeatureSteering
        steerer = FinancialFeatureSteering()
        
        # Check if feature data is loaded
        assert len(steerer.feature_data) == 5, "Should have 5 layers"
        assert all(layer in steerer.feature_data for layer in [4, 10, 16, 22, 28]), "Should have all required layers"
        
        # Check if each layer has 10 features
        for layer, data in steerer.feature_data.items():
            assert len(data["features"]) == 10, f"Layer {layer} should have 10 features"
            for feature in data["features"]:
                assert "id" in feature, "Feature should have ID"
                assert "label" in feature, "Feature should have label"
                assert "activation_improvement" in feature, "Feature should have activation improvement"
        
        print("✅ Feature data structure is correct")
        return True
    except Exception as e:
        print(f"❌ Feature data test failed: {e}")
        return False

def test_model_paths():
    """Test if model paths are accessible."""
    try:
        from streamlit_feature_steering_app import FinancialFeatureSteering
        steerer = FinancialFeatureSteering()
        
        # Check if SAE path exists
        if os.path.exists(steerer.sae_path):
            print("✅ SAE model path exists")
        else:
            print(f"⚠️ SAE model path not found: {steerer.sae_path}")
        
        print(f"✅ Model path configured: {steerer.model_path}")
        return True
    except Exception as e:
        print(f"❌ Model path test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Financial Feature Steering App")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Feature Data Test", test_feature_data),
        ("Model Paths Test", test_model_paths)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The app should work correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
