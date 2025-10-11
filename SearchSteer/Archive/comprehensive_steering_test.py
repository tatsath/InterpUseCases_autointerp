#!/usr/bin/env python3
"""
Comprehensive steering test with multiple features and higher strengths
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open

def test_comprehensive_steering():
    """Test steering with multiple features, higher strengths, and diverse prompts."""
    
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
    
    with safe_open(f"{sae_path}/layers.{layer_idx}/sae.safetensors", framework="pt", device="cpu") as f:
        decoder = f.get_tensor("W_dec")
    
    # Test different features with their labels
    test_features = [
        (159, "Financial Performance and Growth"),
        (258, "Financial market indicators and metrics"),
        (116, "Financial and business-related themes"),
        (345, "Financial performance metrics"),
        (375, "Financial market terminology and jargon")
    ]
    
    # Diverse test prompts
    test_prompts = [
        "The company's quarterly earnings show",
        "Stock market analysis indicates", 
        "Investment opportunities arise when",
        "Financial performance metrics reveal",
        "The business strategy involves",
        "Market trends suggest that",
        "Economic indicators point to",
        "The financial outlook appears"
    ]
    
    print(f"\n🎯 Comprehensive Steering Test - Layer {layer_idx}")
    print("=" * 80)
    
    for feature_idx, feature_label in test_features:
        print(f"\n🔥 Testing Feature {feature_idx}: '{feature_label}'")
        print("=" * 60)
        
        for prompt in test_prompts[:3]:  # Test first 3 prompts per feature
            print(f"\n📝 Prompt: '{prompt}'")
            print("-" * 50)
            
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Test much higher steering strengths
            for strength in [0, 10, 20, 30]:
                print(f"\n--- Strength: {strength} ---")
                
                def steering_hook(module, input, output):
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                    else:
                        hidden_states = output
                    
                    if strength > 0:
                        decoder_device = decoder.to(hidden_states.device)
                        feature_direction = decoder_device[feature_idx, :]
                        feature_direction = feature_direction.unsqueeze(0).unsqueeze(0)
                        
                        # Normalize the feature direction
                        feature_norm = torch.norm(feature_direction)
                        if feature_norm > 0:
                            feature_direction = feature_direction / feature_norm
                        
                        # Apply much stronger steering
                        steering_vector = strength * 1.0 * feature_direction  # Increased from 0.5 to 1.0
                        steered_hidden = hidden_states + steering_vector
                        
                        # Debug info
                        if not hasattr(steering_hook, 'debug_printed'):
                            steering_magnitude = torch.norm(steering_vector).item()
                            print(f"      🎯 Steering: {steering_magnitude:.4f} magnitude")
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
                            max_new_tokens=60,
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
            
            print()  # Empty line between prompts

if __name__ == "__main__":
    test_comprehensive_steering()
