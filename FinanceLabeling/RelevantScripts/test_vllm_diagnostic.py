#!/usr/bin/env python3
"""
VLLM Diagnostic Script
Tests VLLM functionality with different models and configurations
"""

import os
import sys
import asyncio
import torch
from vllm import LLM, SamplingParams
import time

def test_gpu_memory():
    """Test GPU memory availability"""
    print("🔍 GPU Memory Diagnostic")
    print("=" * 50)
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / (1024**3)  # GB
            allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
            cached = torch.cuda.memory_reserved(i) / (1024**3)  # GB
            free = total_memory - allocated
            
            print(f"GPU {i}: {props.name}")
            print(f"  Total: {total_memory:.2f} GB")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Cached: {cached:.2f} GB")
            print(f"  Free: {free:.2f} GB")
            print()
    else:
        print("❌ No CUDA GPUs available")
        return False
    
    return True

def test_simple_vllm():
    """Test basic VLLM functionality with a small model"""
    print("🧪 Testing Basic VLLM Functionality")
    print("=" * 50)
    
    try:
        # Test with a very small model first
        print("Loading GPT-2 model...")
        llm = LLM(
            model="gpt2",
            gpu_memory_utilization=0.1,  # Use only 10% of GPU memory
            max_model_len=512,
            tensor_parallel_size=1,
            enforce_eager=True,  # Disable CUDA graphs for debugging
        )
        
        print("✅ Model loaded successfully!")
        
        # Test simple generation
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=50
        )
        
        prompts = ["Hello, how are you?"]
        print(f"Testing generation with prompt: {prompts[0]}")
        
        outputs = llm.generate(prompts, sampling_params)
        
        for output in outputs:
            print(f"Generated: {output.outputs[0].text}")
        
        print("✅ Basic VLLM test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Basic VLLM test failed: {e}")
        return False

def test_multi_gpu():
    """Test VLLM with multiple GPUs"""
    print("🖥️  Testing Multi-GPU VLLM")
    print("=" * 50)
    
    gpu_count = torch.cuda.device_count()
    print(f"Available GPUs: {gpu_count}")
    
    if gpu_count < 2:
        print("⚠️  Only 1 GPU available, skipping multi-GPU test")
        return True
    
    try:
        # Test with 2 GPUs
        print("Testing with 2 GPUs...")
        llm = LLM(
            model="gpt2",
            gpu_memory_utilization=0.2,  # Use 20% per GPU
            max_model_len=512,
            tensor_parallel_size=2,
            enforce_eager=True,
        )
        
        print("✅ Multi-GPU model loaded successfully!")
        
        # Test generation
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=30
        )
        
        prompts = ["What is machine learning?"]
        outputs = llm.generate(prompts, sampling_params)
        
        for output in outputs:
            print(f"Generated: {output.outputs[0].text}")
        
        print("✅ Multi-GPU test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Multi-GPU test failed: {e}")
        return False

def test_long_sequences():
    """Test VLLM with longer sequences"""
    print("📏 Testing Long Sequence Support")
    print("=" * 50)
    
    try:
        # Test with a model that supports longer sequences
        print("Testing with Qwen2.5-3B-Instruct (supports longer sequences)...")
        
        llm = LLM(
            model="Qwen/Qwen2.5-3B-Instruct",
            gpu_memory_utilization=0.3,
            max_model_len=2048,  # Test with longer sequences
            tensor_parallel_size=1,
            enforce_eager=True,
        )
        
        print("✅ Long sequence model loaded!")
        
        # Test with a longer prompt
        long_prompt = "Explain machine learning in detail. " * 20  # ~500 tokens
        print(f"Testing with long prompt (~{len(long_prompt.split())} words)")
        
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=100
        )
        
        outputs = llm.generate([long_prompt], sampling_params)
        
        for output in outputs:
            print(f"Generated: {output.outputs[0].text[:100]}...")
        
        print("✅ Long sequence test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Long sequence test failed: {e}")
        return False

def test_dialogpt_issues():
    """Test DialoGPT specifically to understand the issues"""
    print("🔍 Testing DialoGPT Issues")
    print("=" * 50)
    
    try:
        print("Testing DialoGPT-small with different configurations...")
        
        # Test 1: Basic DialoGPT
        print("Test 1: Basic DialoGPT with short sequences")
        llm = LLM(
            model="microsoft/DialoGPT-small",
            gpu_memory_utilization=0.2,
            max_model_len=512,  # Stay within model limits
            tensor_parallel_size=1,
            enforce_eager=True,
        )
        
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=50
        )
        
        short_prompt = "Hello, how are you?"
        outputs = llm.generate([short_prompt], sampling_params)
        
        for output in outputs:
            print(f"DialoGPT response: {output.outputs[0].text}")
        
        print("✅ DialoGPT basic test passed!")
        
        # Test 2: Try longer sequences (this should fail)
        print("\nTest 2: DialoGPT with longer sequences (should fail)")
        try:
            long_prompt = "This is a very long prompt. " * 50  # ~150 tokens
            outputs = llm.generate([long_prompt], sampling_params)
            print("⚠️  Long sequence unexpectedly worked")
        except Exception as e:
            print(f"✅ Expected failure with long sequences: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ DialoGPT test failed: {e}")
        return False

def main():
    """Run all diagnostic tests"""
    print("🚀 VLLM Diagnostic Suite")
    print("=" * 60)
    print()
    
    # Set environment variables for better debugging
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"  # Use all GPUs
    
    tests = [
        ("GPU Memory", test_gpu_memory),
        ("Basic VLLM", test_simple_vllm),
        ("Multi-GPU", test_multi_gpu),
        ("Long Sequences", test_long_sequences),
        ("DialoGPT Issues", test_dialogpt_issues),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)
        
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
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! VLLM is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()





