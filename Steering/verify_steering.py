#!/usr/bin/env python3
"""
Verify steering implementation against SAELens approach
"""

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open

def verify_steering_approach():
    """Verify our steering implementation matches SAELens approach."""
    print("🔍 Verifying Steering Implementation")
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
    
    # Load SAE weights
    layer_idx = 22
    feature_idx = 50
    layer_path = os.path.join(sae_path, f"layers.{layer_idx}")
    sae_file = os.path.join(layer_path, "sae.safetensors")
    
    try:
        with safe_open(sae_file, framework="pt", device="cpu") as f:
            encoder = f.get_tensor("encoder.weight")
            encoder_bias = f.get_tensor("encoder.bias")
            decoder = f.get_tensor("W_dec")
            decoder_bias = f.get_tensor("b_dec")
        print(f"✅ SAE weights loaded - Decoder shape: {decoder.shape}")
    except Exception as e:
        print(f"❌ Error loading SAE: {str(e)}")
        return
    
    # Test prompt
    test_prompt = "The business strategy involves"
    inputs = tokenizer(test_prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    print(f"\n📝 Testing prompt: '{test_prompt}'")
    
    # Test different steering approaches
    steering_strengths = [0.0, 5.0, 10.0, 20.0]
    
    for strength in steering_strengths:
        print(f"\n🎛️  Testing strength: {strength}")
        
        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            if strength > 0:
                # Move decoder to same device as hidden states
                decoder_device = decoder.to(hidden_states.device)
                
                # Get the feature direction from decoder (SAELens approach)
                feature_direction = decoder_device[feature_idx, :]  # Shape: [hidden_dim]
                feature_direction = feature_direction.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, hidden_dim]
                
                # Normalize the feature direction
                feature_norm = torch.norm(feature_direction)
                if feature_norm > 0:
                    feature_direction = feature_direction / feature_norm
                
                # Apply steering (SAELens: add scaled feature direction to hidden states)
                steering_vector = strength * 0.5 * feature_direction
                steered_hidden = hidden_states + steering_vector
                
                # Debug info
                if not hasattr(steering_hook, 'debug_printed'):
                    steering_magnitude = torch.norm(steering_vector).item()
                    feature_magnitude = torch.norm(feature_direction).item()
                    print(f"      Feature magnitude: {feature_magnitude:.4f}")
                    print(f"      Steering magnitude: {steering_magnitude:.4f}")
                    print(f"      Hidden states shape: {hidden_states.shape}")
                    print(f"      Feature direction shape: {feature_direction.shape}")
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
                torch.manual_seed(42)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id
                )
                text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract only the new generated text
                prompt_length = len(test_prompt)
                generated_text = text[prompt_length:].strip()
                
                print(f"      Generated: {generated_text}")
                
        except Exception as e:
            print(f"      ❌ Error: {str(e)}")
        finally:
            hook.remove()
    
    print(f"\n🎉 Steering verification complete!")

if __name__ == "__main__":
    verify_steering_approach()
