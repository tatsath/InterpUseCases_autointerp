#!/usr/bin/env python3
"""
Check what weights are available in the SAE safetensors file
"""

import os
from safetensors import safe_open

def check_sae_weights():
    """Check what weights are available in the SAE file."""
    sae_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_finance_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun/layers.22/sae.safetensors"
    
    print("🔍 Checking SAE weights...")
    print(f"File: {sae_path}")
    
    try:
        with safe_open(sae_path, framework="pt", device="cpu") as f:
            keys = f.keys()
            print(f"\n📋 Available keys in SAE file:")
            for key in keys:
                tensor = f.get_tensor(key)
                print(f"   {key}: {tensor.shape} ({tensor.dtype})")
    except Exception as e:
        print(f"❌ Error reading SAE file: {str(e)}")

if __name__ == "__main__":
    check_sae_weights()
