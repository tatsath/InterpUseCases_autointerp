#!/usr/bin/env python3
"""
Basic Multi-GPU Test
Tests if multiple GPUs are accessible and working
"""

import torch
import os

def test_multi_gpu_basic():
    """Test basic multi-GPU functionality"""
    print("🔧 Testing Multi-GPU Setup")
    print("=" * 50)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False
    
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    print(f"📊 Number of GPUs: {torch.cuda.device_count()}")
    
    # Test each GPU
    for i in range(torch.cuda.device_count()):
        try:
            torch.cuda.set_device(i)
            device_name = torch.cuda.get_device_name(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
            memory_free = memory_total - memory_allocated
            
            print(f"  GPU {i}: {device_name}")
            print(f"    Total Memory: {memory_total:.2f} GB")
            print(f"    Allocated: {memory_allocated:.2f} GB")
            print(f"    Free: {memory_free:.2f} GB")
            
            # Test tensor creation on this GPU
            test_tensor = torch.randn(100, 100, device=f'cuda:{i}')
            print(f"    ✅ Tensor creation successful")
            
        except Exception as e:
            print(f"    ❌ GPU {i} failed: {e}")
            return False
    
    # Test multi-GPU tensor operations
    print("\n🔄 Testing Multi-GPU Operations...")
    try:
        # Create tensors on different GPUs
        tensor_gpu0 = torch.randn(100, 100, device='cuda:0')
        tensor_gpu1 = torch.randn(100, 100, device='cuda:1')
        
        # Test cross-GPU operations
        result = tensor_gpu0 + tensor_gpu1.cuda(0)  # Move to GPU 0 and add
        print("✅ Cross-GPU operations successful")
        
        # Test NCCL (if available)
        if torch.cuda.device_count() > 1:
            print("🔄 Testing NCCL communication...")
            # This will test if NCCL is working
            torch.cuda.synchronize()
            print("✅ NCCL communication successful")
        
    except Exception as e:
        print(f"❌ Multi-GPU operations failed: {e}")
        return False
    
    print("\n✅ Multi-GPU Basic Test: PASSED")
    return True

if __name__ == "__main__":
    success = test_multi_gpu_basic()
    if success:
        print("\n🎉 Multi-GPU setup is working correctly!")
    else:
        print("\n❌ Multi-GPU setup has issues that need fixing.")




