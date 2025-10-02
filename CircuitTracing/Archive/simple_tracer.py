"""
Ultra-Simple Circuit Tracer

Just the basics: find which features activate across layers for a given prompt.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open
import time

class SimpleTracer:
    def __init__(self, model_path, sae_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
            device_map=None
        ).to(self.device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load SAEs
        self.saes = self._load_saes(sae_path)
        print(f"Loaded {len(self.saes)} SAE layers")
    
    def _load_saes(self, sae_path):
        """Load SAE weights."""
        saes = {}
        layers = [4, 10, 16, 22, 28]
        
        for layer in layers:
            try:
                sae_file = f"{sae_path}/layers.{layer}/sae.safetensors"
                with safe_open(sae_file, framework="pt") as f:
                    encoder = f.get_tensor("encoder.weight").to(self.device, dtype=torch.float16)
                    encoder_bias = f.get_tensor("encoder.bias").to(self.device, dtype=torch.float16)
                
                saes[layer] = {
                    'encoder': encoder,
                    'bias': encoder_bias
                }
                print(f"✓ Layer {layer}")
            except Exception as e:
                print(f"✗ Layer {layer}: {e}")
        
        return saes
    
    def trace(self, prompt, top_k=10):
        """Simple trace: which features activate for this prompt."""
        print(f"\nTracing: '{prompt[:50]}...'")
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Get SAE features for each layer
        results = {}
        for layer, hidden_states in enumerate(outputs.hidden_states):
            if layer in self.saes:
                # Apply SAE
                sae = self.saes[layer]
                features = torch.matmul(hidden_states, sae['encoder'].T) + sae['bias']
                features = torch.relu(features)
                
                # Get top features
                feature_scores = features.max(dim=1)[0].squeeze()  # Max across sequence
                top_features = torch.topk(feature_scores, top_k)
                
                results[layer] = {
                    'features': top_features.indices.cpu().numpy(),
                    'scores': top_features.values.cpu().numpy()
                }
                
                print(f"Layer {layer}: Top features {top_features.indices.cpu().numpy()[:5]} (scores: {top_features.values.cpu().numpy()[:5]})")
        
        return results
    
    def find_connections(self, results):
        """Find which features appear in multiple layers."""
        print("\n=== FEATURE CONNECTIONS ===")
        
        # Get all features across layers
        all_features = {}
        for layer, data in results.items():
            for i, feature in enumerate(data['features']):
                if feature not in all_features:
                    all_features[feature] = []
                all_features[feature].append((layer, data['scores'][i]))
        
        # Find features that appear in multiple layers
        connections = []
        for feature, appearances in all_features.items():
            if len(appearances) > 1:
                layers = [layer for layer, score in appearances]
                scores = [score for layer, score in appearances]
                connections.append({
                    'feature': int(feature),
                    'layers': layers,
                    'scores': scores,
                    'strength': sum(scores)
                })
        
        # Sort by strength
        connections.sort(key=lambda x: x['strength'], reverse=True)
        
        print(f"Found {len(connections)} features active across multiple layers:")
        for conn in connections[:10]:  # Top 10
            print(f"Feature {conn['feature']}: Layers {conn['layers']} (strength: {conn['strength']:.2f})")
        
        return connections

def main():
    print("="*50)
    print("ULTRA-SIMPLE CIRCUIT TRACER")
    print("="*50)
    
    # Setup
    model_path = "meta-llama/Llama-2-7b-hf"
    sae_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
    
    tracer = SimpleTracer(model_path, sae_path)
    
    # Test prompts
    prompts = [
        "Jamie Dimon leads JPMorgan Chase",
        "Bank earnings rise after rate cuts",
        "Microsoft acquires Activision",
        "Federal Reserve raises rates",
        "Tesla stock increases 15%"
    ]
    
    # Trace each prompt
    all_connections = []
    for i, prompt in enumerate(prompts):
        print(f"\n{'='*20} PROMPT {i+1} {'='*20}")
        results = tracer.trace(prompt)
        connections = tracer.find_connections(results)
        all_connections.extend(connections)
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Analyzed {len(prompts)} prompts")
    print(f"Found {len(all_connections)} total feature connections")
    
    # Most common features
    feature_counts = {}
    for conn in all_connections:
        feature = conn['feature']
        feature_counts[feature] = feature_counts.get(feature, 0) + 1
    
    print(f"\nMost common features across all prompts:")
    for feature, count in sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"Feature {feature}: appears in {count} prompts")
    
    print(f"\n✅ Simple tracing completed!")

if __name__ == "__main__":
    main()
