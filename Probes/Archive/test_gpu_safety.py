#!/usr/bin/env python3
"""
GPU Safety Test Script
Tests the fixed GPU backtesting code to ensure no deadlocks occur
"""

import os
import sys
import torch
import time

# Add paths
sys.path.append('/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Probes')

def test_gpu_safety():
    """Test GPU safety measures"""
    print("="*60)
    print("🧪 GPU SAFETY TEST")
    print("="*60)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    print(f"✅ CUDA available: {torch.cuda.device_count()} GPUs")
    
    # Test GPU IDs
    GPU_IDS = [2, 3, 4, 5]
    
    for gpu_id in GPU_IDS:
        try:
            print(f"\n🧪 Testing GPU {gpu_id}...")
            
            # Clear cache
            torch.cuda.empty_cache()
            
            # Set device
            torch.cuda.set_device(gpu_id)
            torch.cuda.synchronize()
            
            # Test basic operations
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.mm(x, y)
            torch.cuda.synchronize()
            
            # Clear memory
            del x, y, z
            torch.cuda.empty_cache()
            
            print(f"✅ GPU {gpu_id} working correctly")
            
        except Exception as e:
            print(f"❌ GPU {gpu_id} failed: {e}")
            return False
    
    print("\n✅ All GPU safety tests passed!")
    return True

def test_environment_variables():
    """Test environment variable settings"""
    print("\n🧪 Testing environment variables...")
    
    # Check CUDA settings
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    
    # Check VectorBT GPU setting
    vbt_gpu = os.environ.get('VBT_USE_GPU', 'Not set')
    print(f"VBT_USE_GPU: {vbt_gpu}")
    
    # Check CUDA blocking
    cuda_blocking = os.environ.get('CUDA_LAUNCH_BLOCKING', 'Not set')
    print(f"CUDA_LAUNCH_BLOCKING: {cuda_blocking}")
    
    print("✅ Environment variables configured")

def main():
    """Main test function"""
    print("🚀 Starting GPU Safety Tests...")
    
    # Test environment variables
    test_environment_variables()
    
    # Test GPU safety
    if test_gpu_safety():
        print("\n🎉 All tests passed! GPU backtesting should be safe to run.")
        print("💡 You can now run the main backtesting script safely.")
    else:
        print("\n❌ Tests failed! Do not run GPU backtesting.")
        print("💡 Try reducing the number of GPUs or check GPU availability.")

if __name__ == "__main__":
    main()
"""
GPU Safety Test Script
Tests the fixed GPU backtesting code to ensure no deadlocks occur
"""

import os
import sys
import torch
import time

# Add paths
sys.path.append('/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Probes')

def test_gpu_safety():
    """Test GPU safety measures"""
    print("="*60)
    print("🧪 GPU SAFETY TEST")
    print("="*60)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    print(f"✅ CUDA available: {torch.cuda.device_count()} GPUs")
    
    # Test GPU IDs
    GPU_IDS = [2, 3, 4, 5]
    
    for gpu_id in GPU_IDS:
        try:
            print(f"\n🧪 Testing GPU {gpu_id}...")
            
            # Clear cache
            torch.cuda.empty_cache()
            
            # Set device
            torch.cuda.set_device(gpu_id)
            torch.cuda.synchronize()
            
            # Test basic operations
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.mm(x, y)
            torch.cuda.synchronize()
            
            # Clear memory
            del x, y, z
            torch.cuda.empty_cache()
            
            print(f"✅ GPU {gpu_id} working correctly")
            
        except Exception as e:
            print(f"❌ GPU {gpu_id} failed: {e}")
            return False
    
    print("\n✅ All GPU safety tests passed!")
    return True

def test_environment_variables():
    """Test environment variable settings"""
    print("\n🧪 Testing environment variables...")
    
    # Check CUDA settings
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    
    # Check VectorBT GPU setting
    vbt_gpu = os.environ.get('VBT_USE_GPU', 'Not set')
    print(f"VBT_USE_GPU: {vbt_gpu}")
    
    # Check CUDA blocking
    cuda_blocking = os.environ.get('CUDA_LAUNCH_BLOCKING', 'Not set')
    print(f"CUDA_LAUNCH_BLOCKING: {cuda_blocking}")
    
    print("✅ Environment variables configured")

def main():
    """Main test function"""
    print("🚀 Starting GPU Safety Tests...")
    
    # Test environment variables
    test_environment_variables()
    
    # Test GPU safety
    if test_gpu_safety():
        print("\n🎉 All tests passed! GPU backtesting should be safe to run.")
        print("💡 You can now run the main backtesting script safely.")
    else:
        print("\n❌ Tests failed! Do not run GPU backtesting.")
        print("💡 Try reducing the number of GPUs or check GPU availability.")

if __name__ == "__main__":
    main()
"""
GPU Safety Test Script
Tests the fixed GPU backtesting code to ensure no deadlocks occur
"""

import os
import sys
import torch
import time

# Add paths
sys.path.append('/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Probes')

def test_gpu_safety():
    """Test GPU safety measures"""
    print("="*60)
    print("🧪 GPU SAFETY TEST")
    print("="*60)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    print(f"✅ CUDA available: {torch.cuda.device_count()} GPUs")
    
    # Test GPU IDs
    GPU_IDS = [2, 3, 4, 5]
    
    for gpu_id in GPU_IDS:
        try:
            print(f"\n🧪 Testing GPU {gpu_id}...")
            
            # Clear cache
            torch.cuda.empty_cache()
            
            # Set device
            torch.cuda.set_device(gpu_id)
            torch.cuda.synchronize()
            
            # Test basic operations
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.mm(x, y)
            torch.cuda.synchronize()
            
            # Clear memory
            del x, y, z
            torch.cuda.empty_cache()
            
            print(f"✅ GPU {gpu_id} working correctly")
            
        except Exception as e:
            print(f"❌ GPU {gpu_id} failed: {e}")
            return False
    
    print("\n✅ All GPU safety tests passed!")
    return True

def test_environment_variables():
    """Test environment variable settings"""
    print("\n🧪 Testing environment variables...")
    
    # Check CUDA settings
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    
    # Check VectorBT GPU setting
    vbt_gpu = os.environ.get('VBT_USE_GPU', 'Not set')
    print(f"VBT_USE_GPU: {vbt_gpu}")
    
    # Check CUDA blocking
    cuda_blocking = os.environ.get('CUDA_LAUNCH_BLOCKING', 'Not set')
    print(f"CUDA_LAUNCH_BLOCKING: {cuda_blocking}")
    
    print("✅ Environment variables configured")

def main():
    """Main test function"""
    print("🚀 Starting GPU Safety Tests...")
    
    # Test environment variables
    test_environment_variables()
    
    # Test GPU safety
    if test_gpu_safety():
        print("\n🎉 All tests passed! GPU backtesting should be safe to run.")
        print("💡 You can now run the main backtesting script safely.")
    else:
        print("\n❌ Tests failed! Do not run GPU backtesting.")
        print("💡 Try reducing the number of GPUs or check GPU availability.")

if __name__ == "__main__":
    main()
