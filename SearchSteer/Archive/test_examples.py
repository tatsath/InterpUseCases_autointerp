#!/usr/bin/env python3
"""
Test examples for README documentation
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open

def test_steering_examples():
    """Test steering with different prompts and show results."""
    
    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("cxllin/Llama2-7b-Finance")
    model = AutoModelForCausalLM.from_pretrained(
        "cxllin/Llama2-7b-Finance",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load SAE weights
    sae_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_finance_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
    layer_idx = 22
    feature_idx = 159  # "Financial Performance and Growth"
    
    with safe_open(f"{sae_path}/layers.{layer_idx}/sae.safetensors", framework="pt", device="cpu") as f:
        decoder = f.get_tensor("W_dec")
    
    # Test prompts related to financial features
    test_prompts = [
        "The company's quarterly earnings show",
        "Stock market analysis indicates",
        "Investment opportunities arise when",
        "Financial performance metrics reveal",
        "The business strategy involves"
    ]
    
    print(f"\n🎯 Testing Feature {feature_idx} (Financial Performance and Growth)")
    print("=" * 80)
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: '{prompt}'")
        print("-" * 60)
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Test different steering strengths
        for strength in [0, 5, 10, 15]:
            print(f"\n--- Strength: {strength} ---")
            
            def steering_hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                if strength > 0:
                    # Move decoder to same device as hidden_states
                    decoder_device = decoder.to(hidden_states.device)
                    
                    # Get the feature direction from decoder (this is the steering vector)
                    feature_direction = decoder_device[feature_idx, :]  # Shape: [hidden_dim]
                    feature_direction = feature_direction.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, hidden_dim]
                    
                    # Normalize the feature direction for consistent steering
                    feature_norm = torch.norm(feature_direction)
                    if feature_norm > 0:
                        feature_direction = feature_direction / feature_norm
                    
                    # Apply steering: strength * 0.5 * normalized_feature_direction
                    steering_vector = strength * 0.5 * feature_direction
                    
                    # Add steering vector directly to hidden states
                    steered_hidden = hidden_states + steering_vector
                    
                    # Debug info (only print once)
                    if not hasattr(steering_hook, 'debug_printed'):
                        steering_magnitude = torch.norm(steering_vector).item()
                        print(f"      🎯 Steering Implementation:")
                        print(f"         - Feature direction shape: {feature_direction.shape}")
                        print(f"         - Steering magnitude: {steering_magnitude:.4f}")
                        print(f"         - Applied to hidden states: {hidden_states.shape}")
                        steering_hook.debug_printed = True
                else:
                    steered_hidden = hidden_states
                    if hasattr(steering_hook, 'debug_printed'):
                        delattr(steering_hook, 'debug_printed')
                
                if isinstance(output, tuple):
                    return (steered_hidden.to(hidden_states.dtype),) + output[1:]
                else:
                    return steered_hidden.to(hidden_states.dtype)
            
            # Apply steering
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
                        repetition_penalty=1.1,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    generated = text[len(prompt):].strip()
                    print(f"Generated: {generated}")
            finally:
                hook.remove()

if __name__ == "__main__":
    test_steering_examples()
