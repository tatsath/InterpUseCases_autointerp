"""
Simple Circuit Tracer Test

A lightweight test that focuses on the core functionality without
loading the full Llama model to avoid memory and performance issues.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import networkx as nx
from circuit_visualization import CircuitVisualizer, CircuitAnalyzer

def test_visualization_only():
    """Test only the visualization components without model loading."""
    print("Testing visualization components...")
    
    try:
        # Initialize visualizer
        visualizer = CircuitVisualizer()
        
        # Create sample circuit paths for testing
        sample_paths = [
            {
                "path": [(4, 25), (10, 83), (16, 214), (22, 290), (28, 134)],
                "weight_product": 0.85,
                "edge_kinds": ["attn_mediated", "lagged", "attn_mediated", "lagged"]
            },
            {
                "path": [(4, 299), (10, 389), (16, 385), (22, 294), (28, 345)],
                "weight_product": 0.72,
                "edge_kinds": ["lagged", "attn_mediated", "lagged", "attn_mediated"]
            },
            {
                "path": [(4, 32), (10, 91), (16, 279), (22, 258), (28, 276)],
                "weight_product": 0.68,
                "edge_kinds": ["same_layer_corr", "attn_mediated", "lagged", "attn_mediated"]
            }
        ]
        
        # Test circuit path plotting
        print("✓ Creating circuit path visualization...")
        visualizer.plot_circuit_paths(sample_paths, "Sample Circuit Paths", "test_circuit_paths.png")
        
        # Test interactive plot
        print("✓ Creating interactive circuit plot...")
        fig = visualizer.create_interactive_circuit_plot(sample_paths, "Interactive Circuit Paths")
        
        # Save interactive plot
        fig.write_html("test_interactive_circuit.html")
        print("✓ Interactive plot saved to test_interactive_circuit.html")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during visualization testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_graph_construction():
    """Test the graph construction logic without model loading."""
    print("Testing graph construction...")
    
    try:
        # Create mock feature activations
        feats_by_layer = {
            4: torch.randn(1, 10, 400),   # [B, T, F]
            10: torch.randn(1, 10, 400),
            16: torch.randn(1, 10, 400),
            22: torch.randn(1, 10, 400),
            28: torch.randn(1, 10, 400)
        }
        
        # Create mock attention weights
        attn_by_layer = {
            4: torch.randn(1, 32, 10, 10),  # [B, H, T, T]
            10: torch.randn(1, 32, 10, 10),
            16: torch.randn(1, 32, 10, 10),
            22: torch.randn(1, 32, 10, 10)
        }
        
        # Import the build_feature_graph function
        from financial_circuit_tracer import build_feature_graph
        
        # Build feature graph
        print("✓ Building feature graph...")
        G = build_feature_graph(
            feats_by_layer, 
            attn_by_layer,
            attention_weight=0.6,
            lag_weight=0.3,
            same_layer_weight=0.1,
            topk_edges_per_node=5
        )
        
        print(f"✓ Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Test path finding
        from financial_circuit_tracer import k_best_paths
        
        sources = [(4, 25), (4, 299), (4, 32)]
        targets = [(28, 134), (28, 345), (28, 276)]
        
        print("✓ Finding best paths...")
        paths = k_best_paths(G, sources, targets, k=3, max_hops=8)
        
        print(f"✓ Found {len(paths)} circuit paths:")
        for i, path in enumerate(paths):
            path_str = " -> ".join([f"L{layer}:F{feature}" for layer, feature in path["path"]])
            print(f"  {i+1}. [strength={path['weight_product']:.4f}] {path_str}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during graph construction testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_financial_topics():
    """Test the financial topic data structure."""
    print("Testing financial topics...")
    
    try:
        from financial_circuit_tracer import FinancialTopicData
        
        topic_data = FinancialTopicData()
        
        # Test getting all topics
        all_topics = topic_data.get_all_topics()
        print(f"✓ Found {len(all_topics)} financial topics: {all_topics}")
        
        # Test getting prompts for each topic
        for topic in all_topics:
            prompts = topic_data.get_topic_prompts(topic)
            keywords = topic_data.get_topic_keywords(topic)
            print(f"✓ {topic}: {len(prompts)} prompts, {len(keywords)} keywords")
            
            # Show sample prompt
            if prompts:
                print(f"  Sample: {prompts[0][:80]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during financial topics testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all lightweight tests."""
    print("="*60)
    print("SIMPLE CIRCUIT TRACER TEST SUITE")
    print("="*60)
    
    tests = [
        ("Financial Topics Test", test_financial_topics),
        ("Graph Construction Test", test_graph_construction),
        ("Visualization Test", test_visualization_only)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'-'*40}")
        print(f"Running: {test_name}")
        print(f"{'-'*40}")
        
        try:
            success = test_func()
            results[test_name] = "PASSED" if success else "FAILED"
        except Exception as e:
            print(f"Test failed with exception: {str(e)}")
            results[test_name] = "FAILED"
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    for test_name, result in results.items():
        status_icon = "✅" if result == "PASSED" else "❌"
        print(f"{status_icon} {test_name}: {result}")
    
    passed = sum(1 for result in results.values() if result == "PASSED")
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All lightweight tests passed! Core functionality is working.")
        print("Note: Full model testing requires more GPU memory and may need optimization.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
