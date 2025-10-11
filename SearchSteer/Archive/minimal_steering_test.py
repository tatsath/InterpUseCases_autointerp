#!/usr/bin/env python3
"""
Minimal Feature Steering Test
Simple script to test feature steering on financial LLM
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open

def test_steering():
    """Minimal steering test with one feature and one prompt."""
    
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
    feature_idx = 159  # High-impact financial feature
    
    with safe_open(f"{sae_path}/layers.{layer_idx}/sae.safetensors", framework="pt", device="cpu") as f:
        decoder = f.get_tensor("W_dec")
    
    # Test prompt
    prompt = "The business strategy involves"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    print(f"\nTesting prompt: '{prompt}'")
    print(f"Feature: {feature_idx} (Layer {layer_idx})")
    
    # Test different steering strengths
    for strength in [0, 5, 10, 15]:
        print(f"\n--- Strength: {strength} ---")
        
        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            if strength > 0:
                decoder_device = decoder.to(hidden_states.device)
                feature_direction = decoder_device[feature_idx, :].unsqueeze(0).unsqueeze(0)
                feature_norm = torch.norm(feature_direction)
                if feature_norm > 0:
                    feature_direction = feature_direction / feature_norm
                steering_vector = strength * 0.5 * feature_direction
                steered_hidden = hidden_states + steering_vector
            else:
                steered_hidden = hidden_states
            
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
                    max_new_tokens=100,
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
    test_steering()
