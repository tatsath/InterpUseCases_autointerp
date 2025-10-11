#!/usr/bin/env python3
"""
Feature Steering Module

This module handles steering of specific features using SAE weights.
It takes feature IDs from search results and applies steering.
"""

import torch
import os
import time
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open
from typing import Dict, List, Optional

class FeatureSteering:
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf", sae_path: str = None):
        """
        Initialize feature steering
        
        Args:
            model_name: HuggingFace model name or local path
            sae_path: Path to SAE folder (auto-detected if None)
        """
        self.model_name = model_name
        self.sae_path = sae_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.sae_weights = {}
        self.start_time = time.time()
        
        print(f"[{self._get_elapsed_time()}] Initializing FeatureSteering...")
        print(f"[{self._get_elapsed_time()}] Model: {model_name}")
        print(f"[{self._get_elapsed_time()}] Device: {self.device}")
        print(f"[{self._get_elapsed_time()}] SAE Path: {sae_path}")
        
        # Auto-detect SAE path if not provided
        if self.sae_path is None:
            self._setup_default_sae_path()
    
    def _get_elapsed_time(self):
        """Get formatted elapsed time"""
        elapsed = time.time() - self.start_time
        return f"{elapsed:06.1f}s"
    
    def _setup_default_sae_path(self):
        """Setup default SAE path"""
        sae_mapping = {
            "meta-llama/Llama-2-7b-hf": "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun",
            "cxllin/Llama2-7b-Finance": "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_finance_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
        }
        
        self.sae_path = sae_mapping.get(self.model_name, "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun")
        print(f"Using SAE path: {self.sae_path}")
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer"""
        if self.model is None:
            print(f"[{self._get_elapsed_time()}] Loading model and tokenizer...")
            print(f"[{self._get_elapsed_time()}] Loading tokenizer from {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            print(f"[{self._get_elapsed_time()}] Tokenizer loaded successfully")
            
            print(f"[{self._get_elapsed_time()}] Loading model from {self.model_name}...")
            print(f"[{self._get_elapsed_time()}] This may take several minutes for large models...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print(f"[{self._get_elapsed_time()}] Model loaded successfully")
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"[{self._get_elapsed_time()}] Set pad_token to eos_token")
            
            print(f"[{self._get_elapsed_time()}] Model and tokenizer ready!")
    
    def load_sae_weights(self, layer_idx: int):
        """Load SAE weights for specific layer"""
        if layer_idx in self.sae_weights:
            print(f"[{self._get_elapsed_time()}] Using cached SAE weights for layer {layer_idx}")
            return self.sae_weights[layer_idx]
        
        layer_path = os.path.join(self.sae_path, f"layers.{layer_idx}")
        sae_path = os.path.join(layer_path, "sae.safetensors")
        
        print(f"[{self._get_elapsed_time()}] Loading SAE weights for layer {layer_idx}")
        print(f"[{self._get_elapsed_time()}] SAE file: {sae_path}")
        
        if not os.path.exists(sae_path):
            raise FileNotFoundError(f"SAE weights not found at {sae_path}")
        
        print(f"[{self._get_elapsed_time()}] Opening SAE file...")
        with safe_open(sae_path, framework="pt", device="cpu") as f:
            print(f"[{self._get_elapsed_time()}] Loading encoder weights...")
            encoder = f.get_tensor("encoder.weight").to(self.device)
            print(f"[{self._get_elapsed_time()}] Loading encoder bias...")
            encoder_bias = f.get_tensor("encoder.bias").to(self.device)
            print(f"[{self._get_elapsed_time()}] Loading decoder weights...")
            decoder = f.get_tensor("W_dec").to(self.device)
            print(f"[{self._get_elapsed_time()}] Loading decoder bias...")
            decoder_bias = f.get_tensor("b_dec").to(self.device)
        
        self.sae_weights[layer_idx] = (encoder, encoder_bias, decoder, decoder_bias)
        print(f"[{self._get_elapsed_time()}] SAE weights loaded for layer {layer_idx}")
        print(f"[{self._get_elapsed_time()}] Encoder shape: {encoder.shape}")
        print(f"[{self._get_elapsed_time()}] Decoder shape: {decoder.shape}")
        return self.sae_weights[layer_idx]
    
    def steer_feature(self, prompt: str, layer: int, feature_id: int, 
                     steering_strength: float = 10.0, max_tokens: int = 100) -> Dict:
        """Apply feature steering and return results"""
        print(f"[{self._get_elapsed_time()}] Starting feature steering...")
        print(f"[{self._get_elapsed_time()}] Prompt: {prompt}")
        print(f"[{self._get_elapsed_time()}] Layer: {layer}, Feature: {feature_id}")
        print(f"[{self._get_elapsed_time()}] Steering strength: {steering_strength}")
        print(f"[{self._get_elapsed_time()}] Max tokens: {max_tokens}")
        
        self.load_model_and_tokenizer()
        
        # Load SAE weights
        print(f"[{self._get_elapsed_time()}] Loading SAE weights...")
        encoder, encoder_bias, decoder, decoder_bias = self.load_sae_weights(layer)
        
        # Tokenize input
        print(f"[{self._get_elapsed_time()}] Tokenizing input...")
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        print(f"[{self._get_elapsed_time()}] Input tokenized: {inputs['input_ids'].shape}")
        
        # Create steering hook
        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            if abs(steering_strength) > 0.01:
                # Get feature direction from decoder and ensure same device
                feature_direction = decoder[feature_id, :].unsqueeze(0).unsqueeze(0).to(hidden_states.device)
                
                # Normalize
                feature_norm = torch.norm(feature_direction)
                if feature_norm > 0:
                    feature_direction = feature_direction / feature_norm
                
                # Apply steering - using SAELens-style coefficient
                # For +50 intensity, this gives 50 * 0.1 = 5.0 coefficient (more conservative than SAELens 300)
                steering_vector = steering_strength * 0.1 * feature_direction
                steered_hidden = hidden_states + steering_vector
                
                if isinstance(output, tuple):
                    return (steered_hidden.to(hidden_states.dtype),) + output[1:]
                else:
                    return steered_hidden.to(hidden_states.dtype)
            return output
        
        # Register hook
        print(f"[{self._get_elapsed_time()}] Registering steering hook on layer {layer}...")
        layer_module = self.model.model.layers[layer]
        hook = layer_module.register_forward_hook(steering_hook)
        
        try:
            # Generate without steering
            print(f"[{self._get_elapsed_time()}] Generating original text (no steering)...")
            torch.manual_seed(42)
            original_outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            original_text = self.tokenizer.decode(original_outputs[0], skip_special_tokens=True)
            print(f"[{self._get_elapsed_time()}] Original generation completed")
            
            # Generate with steering
            print(f"[{self._get_elapsed_time()}] Generating steered text...")
            torch.manual_seed(42)
            steered_outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=1.0,  # SAELens-style: higher temperature for more creative steering
                top_p=0.1,       # SAELens-style: low top_p for focused generation
                repetition_penalty=1.0,  # SAELens-style: freq_penalty equivalent
                pad_token_id=self.tokenizer.eos_token_id
            )
            steered_text = self.tokenizer.decode(steered_outputs[0], skip_special_tokens=True)
            print(f"[{self._get_elapsed_time()}] Steered generation completed")
            
        finally:
            print(f"[{self._get_elapsed_time()}] Removing steering hook...")
            hook.remove()
            print(f"[{self._get_elapsed_time()}] Hook removed")
        
        return {
            'original_text': original_text,
            'steered_text': steered_text,
            'steering_strength': steering_strength,
            'feature_id': feature_id,
            'layer': layer
        }
    
    def steer_multiple_features(self, prompt: str, features: List[Dict], 
                               steering_strength: float = 10.0, max_tokens: int = 100) -> List[Dict]:
        """Steer multiple features and return results for each"""
        results = []
        
        for feature in features:
            print(f"Steering feature {feature['feature_id']} (Layer {feature['layer']})")
            result = self.steer_feature(
                prompt, 
                feature['layer'], 
                feature['feature_id'], 
                steering_strength, 
                max_tokens
            )
            result['feature_info'] = feature
            results.append(result)
        
        return results
    
    def load_search_results(self, csv_file: str = "1_feature_search_results.csv") -> pd.DataFrame:
        """Load search results from CSV file created by semantic search"""
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Search results file not found: {csv_file}")
        
        print(f"[{self._get_elapsed_time()}] Loading search results from {csv_file}...")
        results_df = pd.read_csv(csv_file)
        print(f"[{self._get_elapsed_time()}] Loaded {len(results_df)} search results")
        return results_df
    
    def get_available_features(self, csv_file: str = "1_feature_search_results.csv") -> List[Dict]:
        """Get list of available features from search results"""
        results_df = self.load_search_results(csv_file)
        
        features = []
        for _, row in results_df.iterrows():
            features.append({
                'feature_id': int(row['feature_id']),
                'feature_name': row['feature_name'],
                'similarity': float(row['similarity']),
                'layer': int(row['layer']),
                'search_keyword': row['search_keyword']
            })
        
        return features
    
    def steer_by_keyword(self, prompt: str, search_keyword: str, 
                        steering_strength: float = 10.0, max_tokens: int = 100,
                        csv_file: str = "1_feature_search_results.csv", 
                        layer: int = 16) -> Dict:
        """
        Steer features based on search keyword
        
        Args:
            prompt: Input text prompt
            search_keyword: Keyword to find features for steering
            steering_strength: Magnitude of steering (positive or negative)
            max_tokens: Maximum tokens to generate
            csv_file: Path to search results CSV
            layer: SAE layer to use for steering (default: 16)
            
        Returns:
            Dict with original and steered text, plus feature info
        """
        print(f"[{self._get_elapsed_time()}] Steering by keyword: '{search_keyword}'")
        
        # Load search results
        results_df = self.load_search_results(csv_file)
        
        # Filter by search keyword and layer
        keyword_results = results_df[
            (results_df['search_keyword'] == search_keyword) & 
            (results_df['layer'] == layer)
        ]
        if keyword_results.empty:
            raise ValueError(f"No features found for keyword: {search_keyword} in layer: {layer}")
        
        print(f"[{self._get_elapsed_time()}] Found {len(keyword_results)} features for '{search_keyword}' in layer {layer}")
        
        # Get the top feature (highest similarity)
        top_feature = keyword_results.loc[keyword_results['similarity'].idxmax()]
        feature_id = int(top_feature['feature_id'])
        actual_layer = int(top_feature['layer'])
        
        print(f"[{self._get_elapsed_time()}] Using top feature: {feature_id} (Layer {actual_layer}, Similarity: {top_feature['similarity']:.3f})")
        
        # Apply steering
        result = self.steer_feature(prompt, actual_layer, feature_id, steering_strength, max_tokens)
        result['search_keyword'] = search_keyword
        result['feature_similarity'] = float(top_feature['similarity'])
        
        return result
    
    def steer_by_feature_id(self, prompt: str, feature_id: int, layer: int,
                           steering_strength: float = 10.0, max_tokens: int = 100) -> Dict:
        """
        Steer by specific feature ID and layer
        
        Args:
            prompt: Input text prompt
            feature_id: Specific feature ID to steer
            layer: SAE layer number
            steering_strength: Magnitude of steering (positive or negative)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dict with original and steered text
        """
        print(f"[{self._get_elapsed_time()}] Steering by feature ID: {feature_id} (Layer {layer})")
        
        result = self.steer_feature(prompt, layer, feature_id, steering_strength, max_tokens)
        result['feature_id'] = feature_id
        result['layer'] = layer
        
        return result
    
    def compare_steering_strengths(self, prompt: str, feature_id: int, layer: int,
                                  strengths: List[float], max_tokens: int = 100) -> List[Dict]:
        """
        Compare different steering strengths for the same feature
        
        Args:
            prompt: Input text prompt
            feature_id: Feature ID to steer
            layer: SAE layer number
            strengths: List of steering strengths to test
            max_tokens: Maximum tokens to generate
            
        Returns:
            List of results for each steering strength
        """
        print(f"[{self._get_elapsed_time()}] Comparing steering strengths: {strengths}")
        
        results = []
        for strength in strengths:
            print(f"[{self._get_elapsed_time()}] Testing strength: {strength}")
            result = self.steer_feature(prompt, layer, feature_id, strength, max_tokens)
            result['steering_strength'] = strength
            results.append(result)
        
        return results

class SteeringUI:
    """
    UI-friendly wrapper for feature steering functionality
    Designed to be called directly from a user interface
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf", sae_path: str = None):
        """Initialize the steering UI wrapper"""
        self.steerer = FeatureSteering(model_name, sae_path)
        print("SteeringUI initialized. Ready for user interactions.")
    
    def get_available_features(self) -> List[Dict]:
        """Get list of available features for UI display"""
        try:
            return self.steerer.get_available_features()
        except FileNotFoundError:
            return []
    
    def steer_text(self, prompt: str, steering_type: str, steering_value: str, 
                  steering_strength: float = 10.0, max_tokens: int = 100) -> Dict:
        """
        Main steering function for UI
        
        Args:
            prompt: Input text to steer
            steering_type: "keyword" or "feature_id"
            steering_value: Keyword to search for or feature ID to use
            steering_strength: Magnitude of steering
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dict with results and metadata
        """
        try:
            if steering_type == "keyword":
                result = self.steerer.steer_by_keyword(
                    prompt=prompt,
                    search_keyword=steering_value,
                    steering_strength=steering_strength,
                    max_tokens=max_tokens
                )
            elif steering_type == "feature_id":
                # Parse feature_id and layer from steering_value (format: "feature_id:layer")
                if ":" in steering_value:
                    feature_id, layer = map(int, steering_value.split(":"))
                else:
                    # Default to layer 16 if not specified
                    feature_id = int(steering_value)
                    layer = 16
                
                result = self.steerer.steer_by_feature_id(
                    prompt=prompt,
                    feature_id=feature_id,
                    layer=layer,
                    steering_strength=steering_strength,
                    max_tokens=max_tokens
                )
            else:
                raise ValueError(f"Invalid steering_type: {steering_type}")
            
            # Add UI-friendly metadata
            result['success'] = True
            result['steering_type'] = steering_type
            result['steering_value'] = steering_value
            result['prompt'] = prompt
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'prompt': prompt,
                'steering_type': steering_type,
                'steering_value': steering_value
            }
    
    def compare_strengths(self, prompt: str, feature_id: int, layer: int, 
                         strengths: List[float], max_tokens: int = 100) -> List[Dict]:
        """Compare different steering strengths for UI"""
        try:
            results = self.steerer.compare_steering_strengths(
                prompt, feature_id, layer, strengths, max_tokens
            )
            for result in results:
                result['success'] = True
            return results
        except Exception as e:
            return [{'success': False, 'error': str(e)}]
    
    def steer_by_feature_id_simple(self, prompt: str, feature_id: int, 
                                  steering_strength: float = 10.0, max_tokens: int = 100) -> Dict:
        """
        Simple method to steer by feature ID (aligned with app usage)
        
        Args:
            prompt: Input text to steer
            feature_id: Feature ID to steer
            steering_strength: Magnitude of steering
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dict with results and metadata
        """
        try:
            result = self.steer_text(
                prompt=prompt,
                steering_type="feature_id",
                steering_value=f"{feature_id}:16",  # Default to layer 16
                steering_strength=steering_strength,
                max_tokens=max_tokens
            )
            return result
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'prompt': prompt,
                'feature_id': feature_id
            }

def main():
    """Example usage"""
    print("=== FEATURE STEERING ===")
    
    # Initialize steering
    steerer = FeatureSteering(
        model_name="meta-llama/Llama-2-7b-hf",
        sae_path="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
    )
    
    prompt = "The bank's credit risk assessment shows"
    
    # Example 1: Load available features from search results
    print("\n1. Loading available features from search results:")
    try:
        available_features = steerer.get_available_features()
        print(f"Found {len(available_features)} available features:")
        for feature in available_features[:3]:  # Show first 3
            print(f"  - Feature {feature['feature_id']}: {feature['feature_name']} (Similarity: {feature['similarity']:.3f})")
    except FileNotFoundError:
        print("No search results found. Run semantic search first.")
        return
    
    # Example 2: Steer by keyword (if search results exist)
    print("\n2. Steering by keyword 'financial performance':")
    try:
        result = steerer.steer_by_keyword(
            prompt=prompt,
            search_keyword="financial performance",
            steering_strength=15.0,
            max_tokens=50,
            layer=16  # Explicitly specify layer
        )
        print(f"Original: {result['original_text'][len(prompt):]}")
        print(f"Steered:  {result['steered_text'][len(prompt):]}")
        print(f"Feature used: {result['feature_id']} (Similarity: {result['feature_similarity']:.3f})")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Example 3: Steer by specific feature ID
    print("\n3. Steering by specific feature ID:")
    result = steerer.steer_by_feature_id(
        prompt=prompt,
        feature_id=339,  # From search results
        layer=16,
        steering_strength=20.0,
        max_tokens=50
    )
    print(f"Original: {result['original_text'][len(prompt):]}")
    print(f"Steered:  {result['steered_text'][len(prompt):]}")
    
    # Example 4: Compare different steering strengths
    print("\n4. Comparing different steering strengths:")
    strengths = [5.0, 15.0, 25.0]
    comparison_results = steerer.compare_steering_strengths(
        prompt=prompt,
        feature_id=339,
        layer=16,
        strengths=strengths,
        max_tokens=30
    )
    
    for result in comparison_results:
        print(f"Strength {result['steering_strength']:2.0f}: {result['steered_text'][len(prompt):]}")
    
    # Example 5: UI Wrapper demonstration
    print("\n5. UI Wrapper demonstration:")
    ui = SteeringUI(
        model_name="meta-llama/Llama-2-7b-hf",
        sae_path="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
    )
    
    # Get available features for UI
    features = ui.get_available_features()
    print(f"Available features for UI: {len(features)}")
    
    # Steer by keyword using UI wrapper
    ui_result = ui.steer_text(
        prompt="The financial analysis indicates",
        steering_type="keyword",
        steering_value="financial performance",
        steering_strength=15.0,
        max_tokens=40
    )
    
    if ui_result['success']:
        print(f"UI Result - Original: {ui_result['original_text'][len(ui_result['prompt']):]}")
        print(f"UI Result - Steered:  {ui_result['steered_text'][len(ui_result['prompt']):]}")
    else:
        print(f"UI Error: {ui_result['error']}")

if __name__ == "__main__":
    main()
