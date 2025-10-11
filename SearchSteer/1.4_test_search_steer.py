#!/usr/bin/env python3
"""
Test script for credit risk search and steering
"""

import importlib.util
import sys

# Import step 1: semantic feature search
spec1 = importlib.util.spec_from_file_location("semantic_search", "1_search_feature.py")
semantic_search = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(semantic_search)
SemanticFeatureSearch = semantic_search.SemanticFeatureSearch

# Import step 2: feature steering  
spec2 = importlib.util.spec_from_file_location("feature_steering", "1.2_steer_feature.py")
feature_steering = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(feature_steering)
FeatureSteering = feature_steering.FeatureSteering
SteeringUI = feature_steering.SteeringUI

def test_medical_features():
    """Test medical feature search and steering"""
    print("=== TESTING MEDICAL FEATURE SEARCH AND STEERING ===")
    print("This process may take 20+ minutes due to model loading...")
    print("=" * 80)
    
    # Step 1: Search for medical features using real labels (like in search_steer_app.py)
    print("\n🔍 STEP 1: SEARCHING FOR 'MEDICAL' FEATURES")
    print("=" * 50)
    
    # Use the same approach as search_steer_app.py
    searcher = SemanticFeatureSearch("llama2_7b_hf", layer=16)
    results = searcher.search_features_with_real_labels("medical", top_k=5)
    
    if not results:
        print("No features found!")
        return
    
    print(f"\n✅ Found {len(results)} features:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. Feature {result['feature_id']} (Layer {result['layer']})")
        print(f"     Label: {result['label']}")
        print(f"     Similarity: {result['similarity']:.3f}")
    
    # Step 2: Apply steering to top feature
    print(f"\n🎯 STEP 2: APPLYING STEERING TO TOP FEATURE")
    print("=" * 50)
    print("⚠️  WARNING: Model loading will take 15-20 minutes...")
    
    # Use SteeringUI for better alignment with the app
    steerer = SteeringUI("meta-llama/Llama-2-7b-hf")
    
    best_feature = results[0]
    prompt = "What are the side effects of aspirin?"
    
    print(f"\n📝 Prompt: {prompt}")
    print(f"🎯 Using Feature {best_feature['feature_id']} (Layer {best_feature['layer']})")
    print(f"🏷️  Feature Label: {best_feature['label']}")
    print()
    
    # Test different steering strengths (aligned with app's range)
    print("🔄 TESTING DIFFERENT STEERING STRENGTHS")
    print("=" * 50)
    for strength in [0, 10, 20, 30, 50]:
        print(f"\n🧪 Testing Steering Strength: {strength}")
        print("-" * 30)
        result = steerer.steer_by_feature_id_simple(
            prompt, 
            best_feature['feature_id'], 
            steering_strength=strength,
            max_tokens=100
        )
        
        if result.get('success', False):
            print(f"📄 Original: {result['original_text'][len(prompt):]}")
            print(f"🎯 Steered:  {result['steered_text'][len(prompt):]}")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
        print("=" * 60)


if __name__ == "__main__":
    test_medical_features()
