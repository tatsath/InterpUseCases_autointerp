"""
Multi-GPU Circuit Tracer Test

Test the circuit tracer with multi-GPU support and reduced memory usage.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from financial_circuit_tracer import FinancialCircuitTracer

def test_multi_gpu_setup():
    """Test the multi-GPU setup without running full analysis."""
    print("Testing multi-GPU setup...")
    
    try:
        # Configuration
        model_path = "meta-llama/Llama-2-7b-hf"
        sae_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
        
        print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        print("Initializing Financial Circuit Tracer with multi-GPU support...")
        tracer = FinancialCircuitTracer(model_path, sae_path, device="cuda")
        
        print(f"Model device mapping:")
        for name, param in tracer.model.named_parameters():
            if 'layers.0' in name or 'layers.4' in name or 'layers.10' in name:
                print(f"  {name}: {param.device}")
        
        print("✓ Multi-GPU setup successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error during multi-GPU setup: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_forward_pass():
    """Test a simple forward pass with minimal memory usage."""
    print("Testing simple forward pass...")
    
    try:
        # Configuration
        model_path = "meta-llama/Llama-2-7b-hf"
        sae_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
        
        print("Initializing tracer...")
        tracer = FinancialCircuitTracer(model_path, sae_path, device="cuda")
        
        # Use a very short prompt to minimize memory usage
        short_prompt = "Jamie Dimon leads JPMorgan."
        
        print(f"Testing with short prompt: '{short_prompt}'")
        
        # Test just the forward pass without full circuit tracing
        toks = tracer.tokenizer([short_prompt], return_tensors="pt", padding=True, truncation=True, max_length=64)
        toks = {k: v.to(tracer.device) for k, v in toks.items()}
        
        print("Running forward pass...")
        with torch.no_grad():
            outputs = tracer.model(**toks, output_hidden_states=True)
        
        print(f"✓ Forward pass successful! Hidden states shape: {outputs.hidden_states[0].shape}")
        
        # Test SAE encoding on one layer
        if len(tracer.sae_wrappers) > 0:
            sae = tracer.sae_wrappers[0]  # Use first SAE
            hidden_states = outputs.hidden_states[5]  # Use layer 5 (after layer 4)
            
            print(f"Testing SAE encoding on layer {sae.layer}...")
            features = sae.encode(hidden_states)
            print(f"✓ SAE encoding successful! Features shape: {features.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during forward pass: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_circuit_tracing_minimal():
    """Test circuit tracing with minimal setup."""
    print("Testing minimal circuit tracing...")
    
    try:
        # Configuration
        model_path = "meta-llama/Llama-2-7b-hf"
        sae_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
        
        print("Initializing tracer...")
        tracer = FinancialCircuitTracer(model_path, sae_path, device="cuda")
        
        # Use a very short prompt
        short_prompt = "Bank earnings rise."
        
        print(f"Testing circuit tracing with: '{short_prompt}'")
        
        # Run circuit tracing with minimal parameters
        circuit_paths, graph, activations = tracer.trace_circuits_for_prompt(
            prompt=short_prompt,
            topic="financial_leaders",
            start_layer=4,
            end_layer=10,  # Use fewer layers
            k_paths=2,     # Fewer paths
            attention_weight=0.5,
            lag_weight=0.3,
            same_layer_weight=0.2
        )
        
        print(f"✓ Circuit tracing successful!")
        print(f"  Found {len(circuit_paths)} circuit paths")
        print(f"  Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        
        # Print circuit paths
        for i, path in enumerate(circuit_paths):
            path_str = " -> ".join([f"L{layer}:F{feature}" for layer, feature in path["path"]])
            print(f"  Path {i+1}: {path_str} (strength: {path['weight_product']:.4f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during circuit tracing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run multi-GPU tests."""
    print("="*60)
    print("MULTI-GPU CIRCUIT TRACER TEST SUITE")
    print("="*60)
    
    tests = [
        ("Multi-GPU Setup Test", test_multi_gpu_setup),
        ("Simple Forward Pass Test", test_simple_forward_pass),
        ("Minimal Circuit Tracing Test", test_circuit_tracing_minimal)
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
        print("🎉 All multi-GPU tests passed! Circuit tracer is ready for production.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
