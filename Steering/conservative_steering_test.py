#!/usr/bin/env python3
"""
SAELens-style steering test with direct feature direction addition
"""

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open

def conservative_steering_test():
    """Test SAELens-style steering approach."""
    print("🧭 SAELens-Style Steering Test")
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
    
    # Test cases - using actual features from the SAE analysis
    test_cases = [
        (22, 159, "Article titles and stock market-related keywords", "The business strategy involves"),
        (22, 258, "Temporal relationships and causal connections", "The business strategy involves"),
        (28, 116, "Prepositional phrases indicating direction", "The future of work includes"),
        (28, 345, "Financial Market and Business Terminology", "The artificial intelligence systems")
    ]
    
    # Additional test prompts for each theme - reduced for faster testing
    additional_prompts = {
        "Article titles and stock market-related keywords": [
            "The revenue growth demonstrates",
            "Stock market analysis shows"
        ],
        "Temporal relationships and causal connections": [
            "The business strategy involves",
            "Market trends indicate"
        ],
        "Prepositional phrases indicating direction": [
            "The future of work includes",
            "Technology development leads to"
        ],
        "Financial Market and Business Terminology": [
            "The artificial intelligence systems",
            "Investment opportunities arise"
        ]
    }
    
    for layer_idx, feature_idx, expected_theme, prompt in test_cases:
        print(f"\n🎯 Testing Layer {layer_idx}, Feature {feature_idx}: {expected_theme}")
        print("=" * 80)
        
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
        
        # Test multiple prompts for this feature
        all_prompts = [prompt] + additional_prompts.get(expected_theme, [])
        
        for test_prompt in all_prompts:
            print(f"\n📝 Testing prompt: '{test_prompt}'")
            
            # Tokenize
            inputs = tokenizer(test_prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Test different steering approaches - including very high strengths
            steering_configs = [
                (0.0, "No steering"),
                (1.0, "Light steering"),
                (2.0, "Moderate steering"),
                (5.0, "Strong steering"),
                (10.0, "Very strong steering")
            ]
            
            # Store outputs for comparison
            outputs_comparison = {}
            
            for strength, description in steering_configs:
                print(f"\n   🎛️  {description} (strength: {strength})")
                
                def conservative_steering_hook(module, input, output):
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                    else:
                        hidden_states = output
                    
                    # Move SAE weights to same device
                    encoder_device = encoder.to(hidden_states.device)
                    encoder_bias_device = encoder_bias.to(hidden_states.device)
                    decoder_device = decoder.to(hidden_states.device)
                    decoder_bias_device = decoder_bias.to(hidden_states.device)
                    
                    # Apply enhanced SAELens-style steering
                    if strength > 0:
                        # Get the feature direction from decoder (this is the steering vector)
                        feature_direction = decoder_device[feature_idx, :]  # Shape: [hidden_dim]
                        feature_direction = feature_direction.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, hidden_dim]
                        
                        # Normalize the feature direction for consistent steering
                        feature_norm = torch.norm(feature_direction)
                        if feature_norm > 0:
                            feature_direction = feature_direction / feature_norm
                        
                        # Use larger steering coefficient for more noticeable effect
                        steering_vector = strength * 0.5 * feature_direction  # Increased from 0.1 to 0.5
                        
                        # Add steering vector directly to hidden states
                        steered_hidden = hidden_states + steering_vector
                        
                        # Debug: Print steering magnitude only once
                        if not hasattr(conservative_steering_hook, 'debug_printed'):
                            steering_magnitude = torch.norm(steering_vector).item()
                            feature_magnitude = torch.norm(feature_direction).item()
                            print(f"      Debug - Feature magnitude: {feature_magnitude:.4f}, Steering magnitude: {steering_magnitude:.4f}, Strength: {strength}")
                            conservative_steering_hook.debug_printed = True
                    else:
                        steered_hidden = hidden_states
                        # Reset debug flag for next test
                        if hasattr(conservative_steering_hook, 'debug_printed'):
                            delattr(conservative_steering_hook, 'debug_printed')
                    
                    if isinstance(output, tuple):
                        return (steered_hidden.to(hidden_states.dtype),) + output[1:]
                    else:
                        return steered_hidden.to(hidden_states.dtype)
                
                # Register hook
                layer_module = model.model.layers[layer_idx]
                hook = layer_module.register_forward_hook(conservative_steering_hook)
                
                try:
                    with torch.no_grad():
                        # Use consistent seed for fair comparison
                        torch.manual_seed(42 + strength)  # Different seed per strength for variety
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=40,  # Reduced for faster testing
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=tokenizer.eos_token_id
                        )
                        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
                        # Analyze the output for theme relevance
                        theme_keywords = {
                            "Article titles and stock market-related keywords": ["stock", "market", "analysis", "trading", "shares", "equity", "portfolio", "investment"],
                            "Temporal relationships and causal connections": ["because", "therefore", "consequently", "leads to", "results in", "causes", "due to", "since"],
                            "Prepositional phrases indicating direction": ["towards", "through", "across", "beyond", "within", "throughout", "during", "while"],
                            "Financial Market and Business Terminology": ["financial", "business", "market", "revenue", "profit", "investment", "capital", "economic"]
                        }
                        
                        keywords = theme_keywords.get(expected_theme, [])
                        matches = [kw for kw in keywords if kw.lower() in text.lower()]
                        relevance_score = len(matches) / len(keywords) if keywords else 0
                        
                        # Extract only the new generated text (after the prompt)
                        prompt_length = len(test_prompt)
                        generated_text = text[prompt_length:].strip()
                        
                        # Store for comparison
                        outputs_comparison[description] = {
                            'text': generated_text,
                            'score': relevance_score,
                            'matches': matches
                        }
                        
                        print(f"      Generated: {generated_text}")
                        print(f"      Theme keywords: {keywords}")
                        print(f"      Matches: {matches} (Score: {relevance_score:.2f})")
                        
                        if relevance_score > 0.3:
                            print(f"      ✅ Good theme alignment!")
                        elif relevance_score > 0.1:
                            print(f"      ⚠️  Moderate theme alignment")
                        else:
                            print(f"      ❌ Poor theme alignment")
                        
                except Exception as e:
                    print(f"      ❌ Error: {str(e)}")
                finally:
                    hook.remove()
            
            # Print comparison summary
            if outputs_comparison:
                print(f"\n   📊 COMPARISON SUMMARY for '{test_prompt}':")
                print("   " + "="*60)
                for desc, data in outputs_comparison.items():
                    print(f"   {desc:20} | Score: {data['score']:.2f} | {data['text'][:50]}...")
                
                # Highlight the difference between no steering and maximum steering
                if "No steering" in outputs_comparison and "Maximum steering" in outputs_comparison:
                    no_steer = outputs_comparison["No steering"]
                    max_steer = outputs_comparison["Maximum steering"]
                    score_diff = max_steer['score'] - no_steer['score']
                    print(f"\n   🎯 HIGH-STRENGTH STEERING IMPACT:")
                    print(f"   Score improvement: {score_diff:+.2f}")
                    if score_diff > 0.1:
                        print(f"   ✅ Significant improvement with high-strength steering!")
                    elif score_diff > 0.05:
                        print(f"   ⚠️  Moderate improvement with high-strength steering")
                    else:
                        print(f"   ❌ Little to no improvement with high-strength steering")
                print()
    
    print(f"\n🎉 SAELens-style steering test complete!")

if __name__ == "__main__":
    conservative_steering_test()
