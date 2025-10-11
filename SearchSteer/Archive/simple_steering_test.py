#!/usr/bin/env python3
"""
Simple steering test - just test if steering has any impact on output
"""

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open

def simple_steering_test():
    """Test basic steering with different feature numbers."""
    print("🧪 Simple Steering Impact Test")
    print("=" * 50)
    
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
    
    # Test different features and layers (SAE has 400 features: 0-399)
    test_configs = [
        (22, 50, "Layer 22, Feature 50"),
        (22, 100, "Layer 22, Feature 100"), 
        (28, 150, "Layer 28, Feature 150"),
        (28, 200, "Layer 28, Feature 200")
    ]
    
    test_prompt = "The business strategy involves"
    
    for layer_idx, feature_idx, description in test_configs:
        print(f"\n🎯 Testing {description}")
        print("=" * 60)
        
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
        
        # Check if feature exists
        if feature_idx >= decoder.shape[0]:
            print(f"   ⚠️  Feature {feature_idx} doesn't exist (max: {decoder.shape[0]-1})")
            continue
            
        # Tokenize
        inputs = tokenizer(test_prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Test different steering strengths
        steering_strengths = [0.0, 1.0, 5.0, 10.0, 20.0]
        outputs = {}
        
        for strength in steering_strengths:
            print(f"\n   🎛️  Strength: {strength}")
            
            def steering_hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                if strength > 0:
                    # Move decoder to same device as hidden_states
                    decoder_device = decoder.to(hidden_states.device)
                    
                    # Get the feature direction from decoder
                    feature_direction = decoder_device[feature_idx, :]  # Shape: [hidden_dim]
                    feature_direction = feature_direction.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, hidden_dim]
                    
                    # Normalize the feature direction
                    feature_norm = torch.norm(feature_direction)
                    if feature_norm > 0:
                        feature_direction = feature_direction / feature_norm
                    
                    # Apply steering
                    steering_vector = strength * 0.5 * feature_direction
                    steered_hidden = hidden_states + steering_vector
                    
                    # Debug info
                    if not hasattr(steering_hook, 'debug_printed'):
                        steering_magnitude = torch.norm(steering_vector).item()
                        print(f"      Debug - Steering magnitude: {steering_magnitude:.4f}")
                        steering_hook.debug_printed = True
                else:
                    steered_hidden = hidden_states
                    if hasattr(steering_hook, 'debug_printed'):
                        delattr(steering_hook, 'debug_printed')
                
                if isinstance(output, tuple):
                    return (steered_hidden.to(hidden_states.dtype),) + output[1:]
                else:
                    return steered_hidden.to(hidden_states.dtype)
            
            # Register hook
            layer_module = model.model.layers[layer_idx]
            hook = layer_module.register_forward_hook(steering_hook)
            
            try:
                with torch.no_grad():
                    torch.manual_seed(42)  # Consistent seed
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=30,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    text = tokenizer.decode(generated[0], skip_special_tokens=True)
                    
                    # Extract only the new generated text
                    prompt_length = len(test_prompt)
                    generated_text = text[prompt_length:].strip()
                    
                    outputs[strength] = generated_text
                    print(f"      Generated: {generated_text}")
                    
            except Exception as e:
                print(f"      ❌ Error: {str(e)}")
            finally:
                hook.remove()
        
        # Compare outputs
        print(f"\n   📊 COMPARISON for {description}:")
        print("   " + "-" * 50)
        for strength, text in outputs.items():
            print(f"   Strength {strength:4.1f}: {text}")
        
        # Check if there are any differences
        unique_outputs = set(outputs.values())
        if len(unique_outputs) > 1:
            print(f"   ✅ STEERING HAS IMPACT! {len(unique_outputs)} different outputs")
        else:
            print(f"   ❌ No steering impact detected")
        print()
    
    print(f"\n🎉 Simple steering test complete!")

if __name__ == "__main__":
    simple_steering_test()
