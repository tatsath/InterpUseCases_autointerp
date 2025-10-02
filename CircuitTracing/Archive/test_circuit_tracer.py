"""
Test script for Financial Circuit Tracer

This script provides a simple way to test the circuit tracing functionality
with sample financial prompts.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from financial_circuit_tracer import FinancialCircuitTracer
from circuit_visualization import CircuitVisualizer, CircuitAnalyzer
import torch

def test_single_prompt():
    """Test circuit tracing with a single financial prompt."""
    print("Testing single prompt circuit tracing...")
    
    # Configuration
    model_path = "meta-llama/Llama-2-7b-hf"
    sae_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    try:
        # Initialize tracer
        print("Initializing Financial Circuit Tracer...")
        tracer = FinancialCircuitTracer(model_path, sae_path, device)
        
        # Test prompt
        test_prompt = "Jamie Dimon's leadership at JPMorgan Chase has been marked by strategic acquisitions and digital transformation initiatives."
        
        print(f"Analyzing prompt: {test_prompt}")
        
        # Trace circuits
        circuit_paths, graph, activations = tracer.trace_circuits_for_prompt(
            prompt=test_prompt,
            topic="financial_leaders",
            start_layer=4,
            end_layer=28,
            k_paths=3
        )
        
        # Print results
        print(f"\nFound {len(circuit_paths)} circuit paths:")
        for i, path in enumerate(circuit_paths):
            path_str = " -> ".join([f"L{layer}:F{feature}" for layer, feature in path["path"]])
            print(f"  {i+1}. [strength={path['weight_product']:.4f}] {path_str}")
            print(f"     Edge types: {path['edge_kinds']}")
        
        # Graph statistics
        print(f"\nGraph statistics:")
        print(f"  Nodes: {graph.number_of_nodes()}")
        print(f"  Edges: {graph.number_of_edges()}")
        
        return True
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_topic_analysis():
    """Test circuit tracing for a specific financial topic."""
    print("\nTesting topic analysis...")
    
    # Configuration
    model_path = "meta-llama/Llama-2-7b-hf"
    sae_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Initialize tracer
        tracer = FinancialCircuitTracer(model_path, sae_path, device)
        
        # Test with mergers and acquisitions topic
        print("Analyzing 'mergers_acquisitions' topic...")
        results = tracer.analyze_topic(
            topic="mergers_acquisitions",
            start_layer=4,
            end_layer=28,
            k_paths=3
        )
        
        # Print summary
        print(f"\nTopic analysis results:")
        for prompt_key, prompt_data in results.items():
            if isinstance(prompt_data, dict) and "circuit_paths" in prompt_data:
                print(f"  {prompt_key}: {len(prompt_data['circuit_paths'])} circuits found")
        
        return True
        
    except Exception as e:
        print(f"Error during topic analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_visualization():
    """Test visualization capabilities."""
    print("\nTesting visualization...")
    
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
            }
        ]
        
        # Test circuit path plotting
        print("Creating circuit path visualization...")
        visualizer.plot_circuit_paths(sample_paths, "Sample Circuit Paths", "test_circuit_paths.png")
        
        # Test interactive plot
        print("Creating interactive circuit plot...")
        fig = visualizer.create_interactive_circuit_plot(sample_paths, "Interactive Circuit Paths")
        
        # Save interactive plot
        fig.write_html("test_interactive_circuit.html")
        print("Interactive plot saved to test_interactive_circuit.html")
        
        return True
        
    except Exception as e:
        print(f"Error during visualization testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("FINANCIAL CIRCUIT TRACER TEST SUITE")
    print("="*60)
    
    tests = [
        ("Single Prompt Test", test_single_prompt),
        ("Topic Analysis Test", test_topic_analysis),
        ("Visualization Test", test_visualization)
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
        print("🎉 All tests passed! Circuit tracer is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
