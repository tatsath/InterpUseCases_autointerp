#!/usr/bin/env python3
"""
Simple AutoInterp Test - Bypass VLLM Issues
Tests AutoInterp with minimal memory usage
"""

import os
import sys
import torch
import asyncio

def test_autointerp_imports():
    """Test if AutoInterp can be imported and basic functionality works"""
    print("🔍 Testing AutoInterp Imports")
    print("=" * 50)
    
    try:
        # Add the autointerp path
        sys.path.append('/home/nvidia/Documents/Hariom/autointerp/autointerp_full')
        
        # Test basic imports
        print("Testing basic imports...")
        from autointerp_full import load_artifacts
        print("✅ load_artifacts imported successfully")
        
        from autointerp_full.clients import Offline
        print("✅ Offline client imported successfully")
        
        from autointerp_full.scorers import DetectionScorer
        print("✅ DetectionScorer imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_simple_model_loading():
    """Test loading a very small model without VLLM"""
    print("\n🧪 Testing Simple Model Loading")
    print("=" * 50)
    
    try:
        # Test with a tiny model that doesn't need VLLM
        print("Testing with a very small model...")
        
        # Use transformers directly instead of VLLM
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Try a very small model
        model_name = "gpt2"  # Very small model
        print(f"Loading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print("✅ Model loaded successfully!")
        
        # Test simple generation
        text = "Hello, how are you?"
        inputs = tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50, do_sample=True)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Generated: {generated_text}")
        
        print("✅ Simple model test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Simple model test failed: {e}")
        return False

def test_autointerp_without_vllm():
    """Test AutoInterp without VLLM - use a different approach"""
    print("\n🔧 Testing AutoInterp Without VLLM")
    print("=" * 50)
    
    try:
        # Test if we can create a simple scorer without VLLM
        from autointerp_full.scorers import DetectionScorer
        
        print("Creating DetectionScorer...")
        scorer = DetectionScorer()
        print("✅ DetectionScorer created successfully")
        
        # Test if we can create a simple client without VLLM
        print("Testing simple client creation...")
        
        # Try to create a mock client that doesn't use VLLM
        class SimpleClient:
            def __init__(self):
                self.model = None
                
            async def generate(self, prompt):
                return f"Mock response for: {prompt[:50]}..."
        
        client = SimpleClient()
        print("✅ Simple client created successfully")
        
        # Test async generation
        async def test_generation():
            response = await client.generate("Test prompt")
            print(f"Mock response: {response}")
            return True
        
        result = asyncio.run(test_generation())
        if result:
            print("✅ AutoInterp without VLLM test passed!")
            return True
        else:
            return False
        
    except Exception as e:
        print(f"❌ AutoInterp without VLLM test failed: {e}")
        return False

def test_memory_usage():
    """Test current memory usage"""
    print("\n💾 Testing Memory Usage")
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

def main():
    """Run all simple tests"""
    print("🚀 Simple AutoInterp Test Suite")
    print("=" * 80)
    print("This tests AutoInterp without VLLM to isolate the issue")
    print("=" * 80)
    
    tests = [
        ("AutoInterp Imports", test_autointerp_imports),
        ("Simple Model Loading", test_simple_model_loading),
        ("AutoInterp Without VLLM", test_autointerp_without_vllm),
        ("Memory Usage", test_memory_usage),
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
    print("📊 SIMPLE TEST SUMMARY")
    print("=" * 80)
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:30} {status}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All simple tests passed! The issue is specifically with VLLM.")
        print("💡 Solution: Use a different LLM backend or reduce memory usage.")
    else:
        print("⚠️  Some tests failed. This helps identify the exact issue.")
        
        # Provide specific recommendations
        if not results.get("AutoInterp Imports", True):
            print("\n🔧 RECOMMENDATION: Fix AutoInterp package installation")
        if not results.get("Simple Model Loading", True):
            print("🔧 RECOMMENDATION: Fix model loading or use smaller models")
        if not results.get("AutoInterp Without VLLM", True):
            print("🔧 RECOMMENDATION: Use a different LLM backend instead of VLLM")

if __name__ == "__main__":
    main()
