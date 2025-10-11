#!/usr/bin/env python3
"""
Quick test to verify steering works with specific features
"""

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open

def quick_steering_test():
    """Quick test of steering with specific features."""
    print("⚡ Quick Steering Test")
    print("=" * 40)
    
    # Model and SAE paths
    model_path = "cxllin/Llama2-7b-Finance"
    sae_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_finance_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
    
    print(f"📦 Loading model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ Model loaded!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return
    
    # Test cases: (layer, feature, expected_theme, test_prompt)
    test_cases = [
        (22, 0, "Technology disruption", "The company is focused on"),
        (22, 1, "Financial success", "The business strategy involves"),
        (28, 0, "AI and automation", "The future of work includes"),
        (16, 2, "Market volatility", "The economic outlook shows")
    ]
    
    for layer_idx, feature_idx, expected_theme, prompt in test_cases:
        print(f"\n🎯 Testing Layer {layer_idx}, Feature {feature_idx}: {expected_theme}")
        print(f"📝 Prompt: '{prompt}'")
        
        # Load SAE weights
        layer_path = os.path.join(sae_path, f"layers.{layer_idx}")
        sae_file = os.path.join(layer_path, "sae.safetensors")
        
        try:
            with safe_open(sae_file, framework="pt", device="cpu") as f:
                encoder = f.get_tensor("encoder.weight")
                encoder_bias = f.get_tensor("encoder.bias")
                decoder = f.get_tensor("W_dec")
                decoder_bias = f.get_tensor("b_dec")
            
            # Move to model device
            model_device = next(model.parameters()).device
            encoder = encoder.to(model_device)
            encoder_bias = encoder_bias.to(model_device)
            decoder = decoder.to(model_device)
            decoder_bias = decoder_bias.to(model_device)
            
        except Exception as e:
            print(f"   ❌ Error loading SAE: {str(e)}")
            continue
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Test steering strengths
        for strength in [0.0, 3.0]:
            print(f"\n   🎛️  Strength: {strength}")
            
            def steering_hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                # Move SAE weights to same device
                encoder_device = encoder.to(hidden_states.device)
                encoder_bias_device = encoder_bias.to(hidden_states.device)
                decoder_device = decoder.to(hidden_states.device)
                decoder_bias_device = decoder_bias.to(hidden_states.device)
                
                # Compute SAE activations
                hidden_states_f32 = hidden_states.float()
                activations = torch.relu(torch.matmul(hidden_states_f32, encoder_device.T) + encoder_bias_device)
                
                # Apply steering
                if strength > 0:
                    steering_vector = torch.zeros_like(activations)
                    steering_vector[:, :, feature_idx] = strength
                    steered_activations = activations + steering_vector
                    steered_hidden = torch.matmul(steered_activations, decoder_device) + decoder_bias_device
                else:
                    steered_hidden = hidden_states
                
                if isinstance(output, tuple):
                    return (steered_hidden.to(hidden_states.dtype),) + output[1:]
                else:
                    return steered_hidden.to(hidden_states.dtype)
            
            # Register hook
            layer_module = model.model.layers[layer_idx]
            hook = layer_module.register_forward_hook(steering_hook)
            
            try:
                with torch.no_grad():
                    torch.manual_seed(42)
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=80,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"      {text}")
                    
            except Exception as e:
                print(f"      ❌ Error: {str(e)}")
            finally:
                hook.remove()
    
    print(f"\n🎉 Quick test complete!")

if __name__ == "__main__":
    quick_steering_test()
