#!/usr/bin/env python3
"""
Simple test to verify steering works without device issues
"""

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open

def test_simple_steering():
    """Test basic steering functionality."""
    print("🧪 Testing Simple Steering")
    print("=" * 40)
    
    # Model and SAE paths
    model_path = "cxllin/Llama2-7b-Finance"
    sae_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_finance_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
    
    print(f"📦 Loading model: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ Model loaded successfully!")
        print(f"   Model device: {next(model.parameters()).device}")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return
    
    # Test text
    test_text = "The stock market is performing well today."
    print(f"\n📝 Test text: '{test_text}'")
    
    # Tokenize
    inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    print(f"   Input device: {inputs['input_ids'].device}")
    
    # Test basic generation without steering
    print(f"\n🔧 Testing basic generation...")
    with torch.no_grad():
        torch.manual_seed(42)
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   Generated: {text}")
    
    # Test SAE loading
    layer_idx = 22
    layer_path = os.path.join(sae_path, f"layers.{layer_idx}")
    sae_file = os.path.join(layer_path, "sae.safetensors")
    
    print(f"\n🔍 Testing SAE loading for layer {layer_idx}...")
    try:
        with safe_open(sae_file, framework="pt", device="cpu") as f:
            encoder = f.get_tensor("encoder.weight")
            bias = f.get_tensor("encoder.bias")
        
        print(f"   ✅ SAE weights loaded - Encoder: {encoder.shape}, Bias: {bias.shape}")
        print(f"   ✅ Encoder device: {encoder.device}, Bias device: {bias.device}")
        
        # Test device movement
        model_device = next(model.parameters()).device
        encoder_moved = encoder.to(model_device)
        bias_moved = bias.to(model_device)
        print(f"   ✅ After moving to model device - Encoder: {encoder_moved.device}, Bias: {bias_moved.device}")
        
    except Exception as e:
        print(f"   ❌ Error loading SAE: {str(e)}")
        return
    
    print(f"\n🎉 Simple steering test complete!")

if __name__ == "__main__":
    test_simple_steering()
