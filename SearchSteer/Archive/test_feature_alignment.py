#!/usr/bin/env python3
"""
Test script to verify feature steering aligns with feature labels
"""

import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open

def load_feature_data():
    """Load feature data from README.md or create test features."""
    # Test features based on common financial/technology themes
    test_features = {
        4: [
            {"id": 0, "label": "Financial performance and earnings", "expected_keywords": ["earnings", "revenue", "profit", "growth", "financial"]},
            {"id": 1, "label": "Market analysis and trends", "expected_keywords": ["market", "trends", "analysis", "trading", "stocks"]},
            {"id": 2, "label": "Technology and innovation", "expected_keywords": ["technology", "innovation", "digital", "software", "tech"]},
            {"id": 3, "label": "Risk and uncertainty", "expected_keywords": ["risk", "uncertainty", "volatility", "challenge", "concern"]},
            {"id": 4, "label": "Investment and capital", "expected_keywords": ["investment", "capital", "funding", "money", "finance"]}
        ],
        10: [
            {"id": 0, "label": "Company operations and business", "expected_keywords": ["company", "business", "operations", "corporate", "firm"]},
            {"id": 1, "label": "Economic conditions", "expected_keywords": ["economic", "economy", "conditions", "recession", "inflation"]},
            {"id": 2, "label": "Technology sector", "expected_keywords": ["technology", "tech", "software", "digital", "AI"]},
            {"id": 3, "label": "Financial metrics", "expected_keywords": ["metrics", "ratios", "performance", "indicators", "measures"]},
            {"id": 4, "label": "Market sentiment", "expected_keywords": ["sentiment", "optimistic", "pessimistic", "confidence", "outlook"]}
        ],
        16: [
            {"id": 0, "label": "Technology innovation", "expected_keywords": ["innovation", "technology", "breakthrough", "advancement", "development"]},
            {"id": 1, "label": "Financial stability", "expected_keywords": ["stability", "stable", "secure", "solid", "reliable"]},
            {"id": 2, "label": "Market volatility", "expected_keywords": ["volatility", "volatile", "fluctuation", "unstable", "turbulent"]},
            {"id": 3, "label": "Growth and expansion", "expected_keywords": ["growth", "expansion", "increase", "rising", "growing"]},
            {"id": 4, "label": "Regulatory environment", "expected_keywords": ["regulatory", "regulation", "compliance", "legal", "policy"]}
        ],
        22: [
            {"id": 0, "label": "Technology disruption", "expected_keywords": ["disruption", "technology", "transform", "revolution", "change"]},
            {"id": 1, "label": "Financial success", "expected_keywords": ["success", "successful", "profitable", "achievement", "accomplishment"]},
            {"id": 2, "label": "Market competition", "expected_keywords": ["competition", "competitive", "rival", "market share", "advantage"]},
            {"id": 3, "label": "Investment opportunities", "expected_keywords": ["opportunities", "investment", "potential", "prospects", "chance"]},
            {"id": 4, "label": "Risk management", "expected_keywords": ["risk", "management", "mitigation", "control", "safety"]}
        ],
        28: [
            {"id": 0, "label": "AI and automation", "expected_keywords": ["AI", "artificial intelligence", "automation", "machine learning", "robotics"]},
            {"id": 1, "label": "Financial forecasting", "expected_keywords": ["forecasting", "prediction", "outlook", "projection", "forecast"]},
            {"id": 2, "label": "Market leadership", "expected_keywords": ["leadership", "leader", "leading", "dominant", "pioneer"]},
            {"id": 3, "label": "Sustainability and ESG", "expected_keywords": ["sustainability", "ESG", "environmental", "green", "sustainable"]},
            {"id": 4, "label": "Global markets", "expected_keywords": ["global", "international", "worldwide", "globalization", "international"]}
        ]
    }
    return test_features

def test_feature_alignment():
    """Test if steering features produces expected outputs."""
    print("🎯 Testing Feature Steering Alignment")
    print("=" * 60)
    
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
    
    # Load feature data
    features = load_feature_data()
    
    # Test prompts
    test_prompts = [
        "The company reported strong quarterly results with",
        "The technology sector is experiencing",
        "Investors are concerned about",
        "The market outlook suggests",
        "The business strategy focuses on"
    ]
    
    # Test different layers and features
    test_cases = [
        (22, 0, "Technology disruption"),  # Layer 22, Feature 0
        (22, 1, "Financial success"),      # Layer 22, Feature 1
        (28, 0, "AI and automation"),      # Layer 28, Feature 0
        (16, 2, "Market volatility"),      # Layer 16, Feature 2
        (10, 2, "Technology sector")       # Layer 10, Feature 2
    ]
    
    for layer_idx, feature_idx, expected_theme in test_cases:
        print(f"\n🔍 Testing Layer {layer_idx}, Feature {feature_idx}: {expected_theme}")
        print("-" * 80)
        
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
        
        # Test each prompt
        for prompt in test_prompts:
            print(f"\n📝 Prompt: '{prompt}'")
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Test different steering strengths
            for strength in [0.0, 2.0, 4.0]:
                print(f"   🎛️  Steering strength: {strength}")
                
                def steering_hook(module, input, output):
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                    else:
                        hidden_states = output
                    
                    # Move SAE weights to same device as hidden states
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
                        
                        # Project back to hidden space using proper decoder
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
                            max_new_tokens=100,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=tokenizer.eos_token_id
                        )
                        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
                        # Analyze if output contains expected keywords
                        expected_keywords = features[layer_idx][feature_idx]["expected_keywords"]
                        keyword_matches = [kw for kw in expected_keywords if kw.lower() in text.lower()]
                        
                        print(f"      Output: {text}")
                        print(f"      Expected keywords: {expected_keywords}")
                        print(f"      Matches: {keyword_matches} ({len(keyword_matches)}/{len(expected_keywords)})")
                        
                        # Score the alignment
                        alignment_score = len(keyword_matches) / len(expected_keywords)
                        if alignment_score > 0.3:
                            print(f"      ✅ Good alignment! Score: {alignment_score:.2f}")
                        elif alignment_score > 0.1:
                            print(f"      ⚠️  Moderate alignment. Score: {alignment_score:.2f}")
                        else:
                            print(f"      ❌ Poor alignment. Score: {alignment_score:.2f}")
                        
                except Exception as e:
                    print(f"      ❌ Error: {str(e)}")
                finally:
                    hook.remove()
    
    print(f"\n🎉 Feature alignment test complete!")

if __name__ == "__main__":
    test_feature_alignment()
