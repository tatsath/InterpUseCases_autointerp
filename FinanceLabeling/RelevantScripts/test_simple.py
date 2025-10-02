#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/nvidia/Documents/Hariom/autointerp/autointerp_full')

from autointerp_full.clients.transformers_fast_client import TransformersFastClient
import asyncio

async def test_model():
    print("🧪 Testing TransformersFastClient...")
    
    try:
        # Test model loading
        client = TransformersFastClient(
            model="meta-llama/Llama-2-7b-chat-hf",
            max_memory=0.7,
            max_model_len=512,  # Small for testing
            num_gpus=1
        )
        
        print("✅ Model loaded successfully!")
        
        # Test generation
        print("🧪 Testing generation...")
        response = await client.generate("Hello, how are you?")
        print(f"✅ Generation successful: {response.text[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_model())
    if success:
        print("🎉 Test passed!")
    else:
        print("💥 Test failed!")
        sys.exit(1)



