#!/usr/bin/env python3
"""
LLM-Only Test Script
Tests just the LLM generation part that's failing in AutoInterp Full
"""

import os
import sys
import asyncio
import torch
from vllm import LLM, SamplingParams
import time

def test_autointerp_llm_config():
    """Test the exact LLM configuration used by AutoInterp Full"""
    print("🧪 Testing AutoInterp LLM Configuration")
    print("=" * 60)
    
    # Set the same environment variables as AutoInterp
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    
    try:
        print("Loading Qwen2.5-3B-Instruct with AutoInterp settings...")
        
        # Use the exact same configuration as AutoInterp
        llm = LLM(
            model="Qwen/Qwen2.5-3B-Instruct",
            gpu_memory_utilization=0.3,  # Same as AutoInterp
            max_model_len=2048,  # Same as AutoInterp
            tensor_parallel_size=2,  # Multi-GPU
            enforce_eager=True,  # Disable CUDA graphs for debugging
        )
        
        print("✅ Model loaded successfully!")
        
        # Test the exact prompt format that AutoInterp uses
        print("\nTesting AutoInterp prompt format...")
        
        # This is the type of prompt that AutoInterp generates
        test_prompts = [
            "Explain what this feature does: Feature 0 shows high activation on financial terms like 'stock', 'market', 'trading'.",
            "What does this feature represent: Feature 1 activates strongly on numerical data and financial calculations.",
            "Describe this feature: Feature 2 is associated with news articles about economic indicators and market analysis."
        ]
        
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=100
        )
        
        print(f"Testing {len(test_prompts)} prompts...")
        
        for i, prompt in enumerate(test_prompts):
            print(f"\nPrompt {i+1}: {prompt[:50]}...")
            
            try:
                outputs = llm.generate([prompt], sampling_params)
                
                for output in outputs:
                    response = output.outputs[0].text
                    print(f"Response: {response[:100]}...")
                
                print(f"✅ Prompt {i+1} successful")
                
            except Exception as e:
                print(f"❌ Prompt {i+1} failed: {e}")
                return False
        
        print("\n✅ All AutoInterp LLM tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ AutoInterp LLM test failed: {e}")
        return False

def test_autointerp_client_simulation():
    """Simulate the exact AutoInterp client usage"""
    print("\n🔧 Testing AutoInterp Client Simulation")
    print("=" * 60)
    
    try:
        # Import the AutoInterp client
        sys.path.append('/home/nvidia/Documents/Hariom/autointerp/autointerp_full')
        from autointerp_full.clients import Offline
        
        print("Testing AutoInterp Offline client...")
        
        # Use the exact same configuration as the script
        client = Offline(
            model="Qwen/Qwen2.5-3B-Instruct",
            max_memory=0.3,
            max_model_len=2048,
            num_gpus=2,
            statistics=True,
        )
        
        print("✅ AutoInterp client created successfully!")
        
        # Test async generation (this is what AutoInterp does)
        async def test_async_generation():
            print("Testing async generation...")
            
            # Test with the same prompt format AutoInterp uses
            test_prompt = "Explain what this feature does: Feature 0 shows high activation on financial terms."
            
            try:
                response = await client.generate(test_prompt)
                print(f"✅ Async generation successful: {response[:100]}...")
                return True
            except Exception as e:
                print(f"❌ Async generation failed: {e}")
                return False
        
        # Run the async test
        result = asyncio.run(test_async_generation())
        
        if result:
            print("✅ AutoInterp client simulation passed!")
            return True
        else:
            return False
        
    except Exception as e:
        print(f"❌ AutoInterp client simulation failed: {e}")
        return False

def test_sequence_length_limits():
    """Test different sequence lengths to find the breaking point"""
    print("\n📏 Testing Sequence Length Limits")
    print("=" * 60)
    
    try:
        llm = LLM(
            model="Qwen/Qwen2.5-3B-Instruct",
            gpu_memory_utilization=0.3,
            max_model_len=2048,
            tensor_parallel_size=1,  # Use single GPU for this test
            enforce_eager=True,
        )
        
        # Test different sequence lengths
        test_lengths = [100, 500, 1000, 1500, 2000]
        
        for length in test_lengths:
            print(f"\nTesting sequence length: {length}")
            
            # Create a prompt of the specified length
            prompt = "This is a test prompt. " * (length // 20)  # Approximate word count
            prompt = prompt[:length]  # Truncate to exact length
            
            try:
                sampling_params = SamplingParams(
                    temperature=0.7,
                    max_tokens=50
                )
                
                outputs = llm.generate([prompt], sampling_params)
                
                for output in outputs:
                    response = output.outputs[0].text
                    print(f"✅ Length {length}: Generated {len(response)} chars")
                
            except Exception as e:
                print(f"❌ Length {length} failed: {e}")
                return False
        
        print("✅ All sequence length tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Sequence length test failed: {e}")
        return False

def main():
    """Run all LLM tests"""
    print("🚀 LLM-Only Test Suite")
    print("=" * 80)
    print("This tests just the LLM part that's failing in AutoInterp Full")
    print("=" * 80)
    
    tests = [
        ("AutoInterp LLM Config", test_autointerp_llm_config),
        ("AutoInterp Client Simulation", test_autointerp_client_simulation),
        ("Sequence Length Limits", test_sequence_length_limits),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*80}")
        print(f"Running: {test_name}")
        print('='*80)
        
        try:
            result = test_func()
            results[test_name] = result
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results[test_name] = False
        
        print()
    
    # Summary
    print("📊 LLM TEST SUMMARY")
    print("=" * 80)
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:30} {status}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All LLM tests passed! The issue is not with the LLM.")
        print("💡 The problem might be in the AutoInterp pipeline or data processing.")
    else:
        print("⚠️  Some LLM tests failed. This identifies the exact issue.")
        
        # Provide specific recommendations
        if not results.get("AutoInterp LLM Config", True):
            print("\n🔧 RECOMMENDATION: Fix the LLM configuration")
        if not results.get("AutoInterp Client Simulation", True):
            print("🔧 RECOMMENDATION: Fix the AutoInterp client usage")
        if not results.get("Sequence Length Limits", True):
            print("🔧 RECOMMENDATION: Reduce sequence length or use a different model")

if __name__ == "__main__":
    main()





