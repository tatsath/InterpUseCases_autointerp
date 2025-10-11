#!/usr/bin/env python3
"""
Debug script to test steering mechanism and identify issues
"""

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open

def debug_steering():
    """Debug the steering mechanism step by step."""
    print("🔍 Debugging Steering Mechanism")
    print("=" * 50)
    
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
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return
    
    # Test text
    test_text = "The company's quarterly earnings show strong growth."
    print(f"\n📝 Test text: '{test_text}'")
    
    # Tokenize
    inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
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
        original_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   Original: {original_text}")
    
    # Load SAE weights
    layer_idx = 22
    layer_path = os.path.join(sae_path, f"layers.{layer_idx}")
    sae_file = os.path.join(layer_path, "sae.safetensors")
    
    print(f"\n🔍 Loading SAE for layer {layer_idx}...")
    try:
        with safe_open(sae_file, framework="pt", device="cpu") as f:
            encoder = f.get_tensor("encoder.weight")
            bias = f.get_tensor("encoder.bias")
        
        print(f"   ✅ SAE weights loaded - Encoder: {encoder.shape}, Bias: {bias.shape}")
        
        # Move to model device
        model_device = next(model.parameters()).device
        encoder = encoder.to(model_device)
        bias = bias.to(model_device)
        
    except Exception as e:
        print(f"   ❌ Error loading SAE: {str(e)}")
        return
    
    # Test different steering approaches
    steering_strengths = [0.0, 1.0, 2.0]
    
    for strength in steering_strengths:
        print(f"\n🎯 Testing steering strength: {strength}")
        
        # Method 1: Simple hidden state modification (no SAE)
        print(f"   Method 1: Simple hidden state modification...")
        try:
            def simple_hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                # Simple steering: add small random noise
                if strength > 0:
                    noise = torch.randn_like(hidden_states) * strength * 0.01
                    steered_hidden = hidden_states + noise
                else:
                    steered_hidden = hidden_states
                
                if isinstance(output, tuple):
                    return (steered_hidden,) + output[1:]
                else:
                    return steered_hidden
            
            # Register hook
            layer_module = model.model.layers[layer_idx]
            hook = layer_module.register_forward_hook(simple_hook)
            
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
                print(f"   Simple steering: {text}")
            
            hook.remove()
            
        except Exception as e:
            print(f"   ❌ Simple steering failed: {str(e)}")
        
        # Method 2: SAE-based steering (current approach)
        print(f"   Method 2: SAE-based steering...")
        try:
            def sae_hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                # Move SAE weights to same device as hidden states
                encoder_device = encoder.to(hidden_states.device)
                bias_device = bias.to(hidden_states.device)
                
                # Compute SAE activations
                hidden_states_f32 = hidden_states.float()
                activations = torch.relu(torch.matmul(hidden_states_f32, encoder_device.T) + bias_device)
                
                # Apply steering
                if strength > 0:
                    steering_vector = torch.zeros_like(activations)
                    steering_vector[:, :, 0] = strength  # Steer first feature
                    steered_activations = activations + steering_vector
                    
                    # Project back to hidden space
                    steered_hidden = torch.matmul(steered_activations, encoder_device)
                else:
                    steered_hidden = hidden_states
                
                if isinstance(output, tuple):
                    return (steered_hidden.to(hidden_states.dtype),) + output[1:]
                else:
                    return steered_hidden.to(hidden_states.dtype)
            
            # Register hook
            layer_module = model.model.layers[layer_idx]
            hook = layer_module.register_forward_hook(sae_hook)
            
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
                print(f"   SAE steering: {text}")
            
            hook.remove()
            
        except Exception as e:
            print(f"   ❌ SAE steering failed: {str(e)}")
    
    print(f"\n🎉 Debug complete!")

if __name__ == "__main__":
    debug_steering()
