#!/usr/bin/env python3
"""
Main Search and Steer Script

This script combines semantic feature search and steering.
It searches for features and then applies steering to the top-k results.
"""

import importlib.util

# Import step 1: semantic feature search
spec1 = importlib.util.spec_from_file_location("semantic_search", "1_semantic_feature_search.py")
semantic_search = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(semantic_search)
SemanticFeatureSearch = semantic_search.SemanticFeatureSearch

# Import step 2: feature steering  
spec2 = importlib.util.spec_from_file_location("feature_steering", "2_feature_steering.py")
feature_steering = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(feature_steering)
FeatureSteering = feature_steering.FeatureSteering
from typing import List, Dict

def search_and_steer(keyword: str, prompt: str, 
                    sae_path: str = "llama2_7b_hf", 
                    layer: int = 16,
                    model_name: str = "meta-llama/Llama-2-7b-hf",
                    steering_strength: float = 10.0, 
                    top_k: int = 3,
                    max_tokens: int = 100) -> List[Dict]:
    """
    Complete search and steer pipeline
    
    Args:
        keyword: Search keyword for features
        prompt: Text prompt for steering
        sae_path: SAE folder path or name
        layer: Layer number to search
        model_name: Model name or path
        steering_strength: Steering strength
        top_k: Number of top features to steer
        max_tokens: Maximum tokens to generate
    
    Returns:
        List of steering results for each feature
    """
    print(f"=== SEARCH AND STEER PIPELINE ===")
    print(f"Keyword: {keyword}")
    print(f"Prompt: {prompt}")
    print(f"SAE Path: {sae_path}")
    print(f"Layer: {layer}")
    print(f"Model: {model_name}")
    print(f"Steering Strength: {steering_strength}")
    print(f"Top-K: {top_k}")
    print()
    
    # Step 1: Search for features
    print("1. Searching for features...")
    searcher = SemanticFeatureSearch(sae_path, layer)
    search_results = searcher.search_features(keyword, top_k)
    
    if not search_results:
        print("No features found!")
        return []
    
    print(f"Found {len(search_results)} features:")
    for i, result in enumerate(search_results, 1):
        print(f"  {i}. Feature {result['feature_id']} (Layer {result['layer']})")
        print(f"     Label: {result['label']}")
        print(f"     Similarity: {result['similarity']:.3f}")
    print()
    
    # Step 2: Apply steering
    print("2. Applying steering...")
    steerer = FeatureSteering(model_name, searcher.sae_path)
    steering_results = steerer.steer_multiple_features(
        prompt, search_results, steering_strength, max_tokens
    )
    
    return steering_results

def main():
    """Example usage with credit risk"""
    print("=== CREDIT RISK SEARCH AND STEER ===")
    
    # Example 1: Credit risk search and steer
    results = search_and_steer(
        keyword="credit risk",
        prompt="The bank's credit risk assessment shows",
        sae_path="llama2_7b_hf",  # Can be SAE name or folder path
        layer=16,
        model_name="meta-llama/Llama-2-7b-hf",
        steering_strength=20.0,
        top_k=3,
        max_tokens=80
    )
    
    # Display results
    print("\n=== STEERING RESULTS ===")
    for i, result in enumerate(results, 1):
        feature = result['feature_info']
        print(f"\n{i}. Feature {feature['feature_id']} (Layer {feature['layer']})")
        print(f"   Label: {feature['label']}")
        print(f"   Similarity: {feature['similarity']:.3f}")
        print(f"   Steering Strength: {result['steering_strength']}")
        print(f"   Original: {result['original_text'][len('The bank\'s credit risk assessment shows'):]}")
        print(f"   Steered:  {result['steered_text'][len('The bank\'s credit risk assessment shows'):]}")
        print("-" * 80)
    
    # Example 2: Different keyword
    print("\n\n=== FINANCIAL PERFORMANCE SEARCH AND STEER ===")
    results2 = search_and_steer(
        keyword="financial performance",
        prompt="The company's quarterly earnings show",
        sae_path="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun",  # Direct path
        layer=16,
        steering_strength=15.0,
        top_k=2
    )
    
    print(f"\nFound {len(results2)} results for financial performance")

if __name__ == "__main__":
    main()
