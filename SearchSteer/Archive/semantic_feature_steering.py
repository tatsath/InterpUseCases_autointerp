#!/usr/bin/env python3
"""
Minimal Semantic Feature Search and Steering Script

This script provides:
1. Semantic search for features from results_summary_layer16.csv
2. Feature steering using SAE weights and Llama-2-7b-hf model
"""

import pandas as pd
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open
import os
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity

class SemanticFeatureSteering:
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf", sae_name="llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"):
        self.model_name = model_name
        self.sae_name = sae_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.features_df = None
        self.model = None
        self.tokenizer = None
        self.sae_weights = {}
        
        # Auto-detect paths and load features
        self._setup_paths()
        self._load_default_features()
    
    def _setup_paths(self):
        """Setup model and SAE paths based on names"""
        # Model path - check if local or huggingface
        if os.path.exists(self.model_name):
            self.model_path = self.model_name
        else:
            self.model_path = self.model_name
            
        # SAE path mapping
        sae_mapping = {
            "llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun": "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun",
            "llama2_7b_finance_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun": "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_finance_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun",
            "llama2_7b_hf": "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun",
            "llama2_7b_finance": "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_finance_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
        }
        
        self.sae_path = sae_mapping.get(self.sae_name, f"/home/nvidia/Documents/Hariom/saetrain/trained_models/{self.sae_name}")
        
        # Feature file mapping
        self.feature_files = {
            "llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun": "results_summary_layer16.csv",
            "llama2_7b_finance_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun": "results_summary_layer16.csv"
        }
        
    def _load_default_features(self):
        """Load features dynamically from SAE files"""
        # Try to load from CSV first, then fallback to SAE discovery
        feature_file = self.feature_files.get(self.sae_name, "results_summary_layer16.csv")
        if os.path.exists(feature_file):
            self.features_df = pd.read_csv(feature_file)
            print(f"Auto-loaded {len(self.features_df)} features from {feature_file}")
        else:
            # Discover features from SAE files
            self._discover_sae_features()
    
    def _discover_sae_features(self):
        """Discover features by loading SAE weights and creating feature labels"""
        print("Discovering features from SAE files...")
        
        # Try different layers to find available SAE files
        available_layers = []
        for layer in [4, 10, 16, 22, 28]:
            layer_path = os.path.join(self.sae_path, f"layers.{layer}")
            sae_path = os.path.join(layer_path, "sae.safetensors")
            if os.path.exists(sae_path):
                available_layers.append(layer)
        
        if not available_layers:
            print("No SAE files found!")
            return
        
        print(f"Found SAE files for layers: {available_layers}")
        
        # Load features from layer 16 (or first available)
        target_layer = 16 if 16 in available_layers else available_layers[0]
        self._load_sae_features_for_layer(target_layer)
    
    def _load_sae_features_for_layer(self, layer: int):
        """Load SAE features for a specific layer and create feature labels"""
        layer_path = os.path.join(self.sae_path, f"layers.{layer}")
        sae_path = os.path.join(layer_path, "sae.safetensors")
        
        if not os.path.exists(sae_path):
            print(f"SAE file not found for layer {layer}")
            return
        
        try:
            with safe_open(sae_path, framework="pt", device="cpu") as f:
                decoder = f.get_tensor("W_dec")
                num_features = decoder.shape[0]
                print(f"Found {num_features} features in layer {layer}")
                
                # Create feature data with generic labels
                features_data = []
                for i in range(num_features):
                    features_data.append({
                        'layer': layer,
                        'feature': i,
                        'label': f"Feature {i} (Layer {layer})",
                        'f1_score': 0.0  # Unknown, will be updated if CSV available
                    })
                
                self.features_df = pd.DataFrame(features_data)
                print(f"Created {len(self.features_df)} feature entries for layer {layer}")
                
        except Exception as e:
            print(f"Error loading SAE features: {e}")
            # Fallback to CSV if available
            if os.path.exists("results_summary_layer16.csv"):
                self.features_df = pd.read_csv("results_summary_layer16.csv")
                print(f"Fallback: Loaded {len(self.features_df)} features from CSV")
    
    def load_features(self, csv_path: str = None):
        """Load features from CSV file"""
        if csv_path is None:
            csv_path = self.feature_files.get(self.sae_name, "results_summary_layer16.csv")
        self.features_df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.features_df)} features from {csv_path}")
        
    def search_features(self, keyword: str, top_k: int = 5) -> List[Dict]:
        """Semantic search for features based on keyword"""
        if self.features_df is None:
            raise ValueError("Features not loaded. Call load_features() first.")
            
        # Get embeddings for all feature labels
        feature_labels = self.features_df['label'].tolist()
        feature_embeddings = self.semantic_model.encode(feature_labels)
        
        # Get embedding for search keyword
        keyword_embedding = self.semantic_model.encode([keyword])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(keyword_embedding, feature_embeddings)[0]
        
        # Get top-k most similar features
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'feature_id': self.features_df.iloc[idx]['feature'],
                'label': self.features_df.iloc[idx]['label'],
                'f1_score': self.features_df.iloc[idx]['f1_score'],
                'similarity': similarities[idx],
                'layer': self.features_df.iloc[idx]['layer']
            })
            
        return results
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer"""
        if self.model is None:
            print("Loading model and tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print("Model and tokenizer loaded successfully!")
    
    def load_sae_weights(self, layer_idx: int):
        """Load SAE weights for specific layer"""
        if layer_idx in self.sae_weights:
            return self.sae_weights[layer_idx]
            
        layer_path = os.path.join(self.sae_path, f"layers.{layer_idx}")
        sae_path = os.path.join(layer_path, "sae.safetensors")
        
        if not os.path.exists(sae_path):
            raise FileNotFoundError(f"SAE weights not found at {sae_path}")
            
        with safe_open(sae_path, framework="pt", device="cpu") as f:
            encoder = f.get_tensor("encoder.weight").to(self.device)
            encoder_bias = f.get_tensor("encoder.bias").to(self.device)
            decoder = f.get_tensor("W_dec").to(self.device)
            decoder_bias = f.get_tensor("b_dec").to(self.device)
            
        self.sae_weights[layer_idx] = (encoder, encoder_bias, decoder, decoder_bias)
        print(f"Loaded SAE weights for layer {layer_idx}")
        return self.sae_weights[layer_idx]
    
    def steer_feature(self, prompt: str, layer: int, feature_id: int, 
                     steering_strength: float = 10.0, max_tokens: int = 100) -> Dict:
        """Apply feature steering and return results"""
        self.load_model_and_tokenizer()
        
        # Load SAE weights
        encoder, encoder_bias, decoder, decoder_bias = self.load_sae_weights(layer)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Create steering hook
        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            if abs(steering_strength) > 0.01:
                # Get feature direction from decoder
                feature_direction = decoder[feature_id, :].unsqueeze(0).unsqueeze(0)
                
                # Normalize
                feature_norm = torch.norm(feature_direction)
                if feature_norm > 0:
                    feature_direction = feature_direction / feature_norm
                
                # Apply steering
                steering_vector = steering_strength * 0.5 * feature_direction
                steered_hidden = hidden_states + steering_vector
                
                if isinstance(output, tuple):
                    return (steered_hidden.to(hidden_states.dtype),) + output[1:]
                else:
                    return steered_hidden.to(hidden_states.dtype)
            return output
        
        # Register hook
        layer_module = self.model.model.layers[layer]
        hook = layer_module.register_forward_hook(steering_hook)
        
        try:
            # Generate without steering
            torch.manual_seed(42)
            original_outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            original_text = self.tokenizer.decode(original_outputs[0], skip_special_tokens=True)
            
            # Generate with steering
            torch.manual_seed(42)
            steered_outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            steered_text = self.tokenizer.decode(steered_outputs[0], skip_special_tokens=True)
            
        finally:
            hook.remove()
        
        return {
            'original_text': original_text,
            'steered_text': steered_text,
            'steering_strength': steering_strength,
            'feature_id': feature_id,
            'layer': layer
        }

def search_and_steer(keyword: str, prompt: str, model_name="meta-llama/Llama-2-7b-hf", 
                     sae_name="llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun",
                     steering_strength=10.0, top_k=3):
    """One-liner function for search and steer"""
    steerer = SemanticFeatureSteering(model_name, sae_name)
    results = steerer.search_features(keyword, top_k)
    if results:
        best = results[0]
        return steerer.steer_feature(prompt, best['layer'], best['feature_id'], steering_strength)
    return None

def main():
    """Example usage with credit risk keyword"""
    print("=== SEMANTIC FEATURE SEARCH & STEERING ===")
    print("Searching for 'credit risk' features in SAE files...")
    
    # Initialize with layer 16 focus
    steerer = SemanticFeatureSteering(
        model_name="meta-llama/Llama-2-7b-hf",
        sae_name="llama2_7b_hf"  # This will auto-discover SAE files
    )
    
    # Search for credit risk features
    print("\n=== SEARCHING FOR CREDIT RISK FEATURES ===")
    results = steerer.search_features("credit risk", top_k=5)
    
    if results:
        print(f"Found {len(results)} features related to 'credit risk':")
        for i, result in enumerate(results, 1):
            print(f"{i}. Feature {result['feature_id']} (Layer {result['layer']})")
            print(f"   Label: {result['label']}")
            print(f"   Similarity: {result['similarity']:.3f}")
            print()
        
        # Test steering with the best feature
        print("=== APPLYING CREDIT RISK STEERING ===")
        best_feature = results[0]
        prompt = "The bank's credit risk assessment shows"
        
        print(f"Prompt: {prompt}")
        print(f"Using Feature {best_feature['feature_id']} (Layer {best_feature['layer']})")
        print(f"Feature Label: {best_feature['label']}")
        print()
        
        # Test different steering strengths
        for strength in [0, 10, 20, 30]:
            result = steerer.steer_feature(
                prompt, 
                best_feature['layer'], 
                best_feature['feature_id'], 
                steering_strength=strength,
                max_tokens=80
            )
            
            print(f"Steering Strength: {strength}")
            print(f"Original: {result['original_text'][len(prompt):]}")
            print(f"Steered:  {result['steered_text'][len(prompt):]}")
            print("-" * 60)
    else:
        print("No features found for 'credit risk'")
        
        # Show available features
        if steerer.features_df is not None:
            print(f"\nAvailable features: {len(steerer.features_df)}")
            print("First 5 features:")
            for i in range(min(5, len(steerer.features_df))):
                print(f"  {i}: {steerer.features_df.iloc[i]['label']}")

if __name__ == "__main__":
    main()
