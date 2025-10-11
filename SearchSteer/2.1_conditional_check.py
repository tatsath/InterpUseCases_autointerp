#!/usr/bin/env python3
"""
Conditional Check Module

This module provides simple functions for feature activation monitoring and conditional logic.
It encapsulates the logic from feature_conditional_app.py into reusable functions.
"""

import torch
import numpy as np
import os
import time
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class ConditionalChecker:
    """Handles feature activation monitoring and conditional logic"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf", sae_path: str = None):
        """
        Initialize conditional checker
        
        Args:
            model_name: HuggingFace model name
            sae_path: Path to SAE folder (auto-detected if None)
        """
        self.model_name = model_name
        self.sae_path = sae_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.semantic_model = None
        self.start_time = time.time()
        
        print(f"[{self._get_elapsed_time()}] Initializing ConditionalChecker...")
        print(f"[{self._get_elapsed_time()}] Model: {model_name}")
        print(f"[{self._get_elapsed_time()}] Device: {self.device}")
        
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
        """Load model and tokenizer if not already loaded"""
        if self.model is None:
            print(f"[{self._get_elapsed_time()}] Loading model and tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"[{self._get_elapsed_time()}] Model and tokenizer loaded")
    
    def load_semantic_model(self):
        """Load semantic model for feature search"""
        if self.semantic_model is None:
            print(f"[{self._get_elapsed_time()}] Loading semantic model...")
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            print(f"[{self._get_elapsed_time()}] Semantic model loaded")
    
    def load_sae_weights(self, layer_idx: int):
        """Load SAE weights for specific layer"""
        layer_path = os.path.join(self.sae_path, f"layers.{layer_idx}")
        sae_path = os.path.join(layer_path, "sae.safetensors")
        
        if not os.path.exists(sae_path):
            raise FileNotFoundError(f"SAE weights not found at {sae_path}")
        
        print(f"[{self._get_elapsed_time()}] Loading SAE weights for layer {layer_idx}")
        with safe_open(sae_path, framework="pt", device="cpu") as f:
            encoder = f.get_tensor("encoder.weight").to(self.device)
            encoder_bias = f.get_tensor("encoder.bias").to(self.device)
            decoder = f.get_tensor("W_dec").to(self.device)
            decoder_bias = f.get_tensor("b_dec").to(self.device)
        
        return encoder, encoder_bias, decoder, decoder_bias
    
    def get_active_features(self, prompt: str, layer: int = 16, top_k: int = 10) -> List[Dict]:
        """
        Get the top K most active features for a given prompt
        
        Args:
            prompt: Input text prompt
            layer: SAE layer to analyze
            top_k: Number of top features to return
            
        Returns:
            List of dictionaries with feature info and activation values
        """
        print(f"[{self._get_elapsed_time()}] Getting active features for prompt...")
        
        self.load_model_and_tokenizer()
        encoder, encoder_bias, decoder, decoder_bias = self.load_sae_weights(layer)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get hidden states from the specified layer
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer]  # Get hidden states from specific layer
        
        # Calculate activations for all features
        num_features = decoder.shape[0]
        activations = []
        
        for feature_id in range(num_features):
            # Get feature direction
            feature_direction = decoder[feature_id, :].to(hidden_states.device)
            
            # Compute activation (dot product with hidden states)
            # Shape: [batch, seq_len, hidden_dim] @ [hidden_dim] -> [batch, seq_len]
            feature_activations = torch.sum(hidden_states * feature_direction, dim=-1)
            
            # Get max activation across sequence
            max_activation = torch.max(feature_activations).item()
            
            activations.append({
                'feature_id': feature_id,
                'max_activation': max_activation,
                'activation_percentage': (max_activation / 100.0) * 100  # Heuristic normalization
            })
        
        # Sort by activation and return top K
        activations.sort(key=lambda x: x['max_activation'], reverse=True)
        
        print(f"[{self._get_elapsed_time()}] Found {len(activations)} features, returning top {top_k}")
        return activations[:top_k]
    
    def check_condition(self, prompt: str, feature_id: int, layer: int = 16, 
                       operator: str = "greater_than", threshold: float = 0.0, 
                       use_percentage: bool = False) -> Dict:
        """
        Check if a specific feature meets a condition
        
        Args:
            prompt: Input text prompt
            feature_id: Feature ID to check
            layer: SAE layer to analyze
            operator: "greater_than" or "less_than"
            threshold: Threshold value to compare against
            use_percentage: Whether to use percentage (0-100) or raw activation
            
        Returns:
            Dictionary with condition result and activation info
        """
        print(f"[{self._get_elapsed_time()}] Checking condition for feature {feature_id}...")
        
        self.load_model_and_tokenizer()
        encoder, encoder_bias, decoder, decoder_bias = self.load_sae_weights(layer)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get hidden states from the specified layer
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer]
        
        # Calculate activation for specific feature
        feature_direction = decoder[feature_id, :].to(hidden_states.device)
        feature_activations = torch.sum(hidden_states * feature_direction, dim=-1)
        max_activation = torch.max(feature_activations).item()
        activation_percentage = (max_activation / 100.0) * 100
        
        # Determine comparison value
        comparison_value = activation_percentage if use_percentage else max_activation
        
        # Check condition
        condition_met = False
        if operator == "greater_than":
            condition_met = comparison_value > threshold
        elif operator == "less_than":
            condition_met = comparison_value < threshold
        else:
            raise ValueError(f"Invalid operator: {operator}. Use 'greater_than' or 'less_than'")
        
        result = {
            'condition_met': condition_met,
            'feature_id': feature_id,
            'layer': layer,
            'max_activation': max_activation,
            'activation_percentage': activation_percentage,
            'comparison_value': comparison_value,
            'threshold': threshold,
            'operator': operator,
            'use_percentage': use_percentage
        }
        
        print(f"[{self._get_elapsed_time()}] Condition result: {condition_met} (activation: {comparison_value:.3f} {operator} {threshold})")
        return result
    
    def check_multiple_conditions(self, prompt: str, conditions: List[Dict], 
                                 layer: int = 16, logic_type: str = "AND") -> Dict:
        """
        Check multiple conditions with AND/OR logic
        
        Args:
            prompt: Input text prompt
            conditions: List of condition dictionaries with keys:
                - feature_id: Feature ID to check
                - operator: "greater_than" or "less_than"
                - threshold: Threshold value
                - use_percentage: Whether to use percentage
            layer: SAE layer to analyze
            logic_type: "AND" or "OR" logic
            
        Returns:
            Dictionary with overall result and individual condition results
        """
        print(f"[{self._get_elapsed_time()}] Checking {len(conditions)} conditions with {logic_type} logic...")
        
        individual_results = []
        met_conditions = []
        failed_conditions = []
        
        # Check each condition individually
        for condition in conditions:
            result = self.check_condition(
                prompt=prompt,
                feature_id=condition['feature_id'],
                layer=layer,
                operator=condition['operator'],
                threshold=condition['threshold'],
                use_percentage=condition.get('use_percentage', False)
            )
            
            individual_results.append(result)
            
            if result['condition_met']:
                met_conditions.append(result)
            else:
                failed_conditions.append(result)
        
        # Apply AND/OR logic
        if logic_type == "AND":
            overall_met = len(met_conditions) == len(conditions)
        elif logic_type == "OR":
            overall_met = len(met_conditions) > 0
        else:
            raise ValueError(f"Invalid logic_type: {logic_type}. Use 'AND' or 'OR'")
        
        result = {
            'overall_met': overall_met,
            'logic_type': logic_type,
            'total_conditions': len(conditions),
            'met_conditions': len(met_conditions),
            'failed_conditions': len(failed_conditions),
            'individual_results': individual_results,
            'met_conditions_list': met_conditions,
            'failed_conditions_list': failed_conditions
        }
        
        print(f"[{self._get_elapsed_time()}] Overall result: {overall_met} ({len(met_conditions)}/{len(conditions)} conditions met)")
        return result
    
    def search_features_by_prompt(self, prompt: str, layer: int = 16, 
                                 top_k: int = 10, keyword: str = None) -> List[Dict]:
        """
        Search for features that are most active for a given prompt
        
        Args:
            prompt: Input text prompt
            layer: SAE layer to analyze
            top_k: Number of top features to return
            keyword: Optional keyword to filter features (requires semantic search)
            
        Returns:
            List of dictionaries with feature info and activation values
        """
        print(f"[{self._get_elapsed_time()}] Searching features for prompt...")
        
        # Get active features
        active_features = self.get_active_features(prompt, layer, top_k)
        
        # If keyword is provided, filter by semantic similarity
        if keyword:
            print(f"[{self._get_elapsed_time()}] Filtering by keyword: {keyword}")
            self.load_semantic_model()
            
            # Load feature labels (you may need to adjust this path)
            try:
                csv_path = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/FinanceLabeling/multi_layer_full_results/multi_layer_full_layer16/results_summary.csv"
                if os.path.exists(csv_path):
                    import pandas as pd
                    df = pd.read_csv(csv_path)
                    feature_labels = {}
                    for _, row in df.iterrows():
                        if row['layer'] == layer:
                            feature_labels[row['feature']] = row['label']
                    
                    # Filter active features by semantic similarity
                    filtered_features = []
                    print(f"[{self._get_elapsed_time()}] Checking {len(active_features)} active features against {len(feature_labels)} labels...")
                    
                    for feature in active_features:
                        if feature['feature_id'] in feature_labels:
                            label = feature_labels[feature['feature_id']]
                            
                            # Calculate semantic similarity
                            label_embedding = self.semantic_model.encode([label])
                            keyword_embedding = self.semantic_model.encode([keyword])
                            similarity = cosine_similarity(keyword_embedding, label_embedding)[0][0]
                            
                            print(f"[{self._get_elapsed_time()}] Feature {feature['feature_id']}: '{label}' vs '{keyword}' = {similarity:.3f}")
                            
                            if similarity > 0.2:  # Lower threshold for relevance
                                feature['label'] = label
                                feature['similarity'] = similarity
                                filtered_features.append(feature)
                    
                    print(f"[{self._get_elapsed_time()}] Found {len(filtered_features)} features matching keyword")
                    return filtered_features[:top_k]
            except Exception as e:
                print(f"[{self._get_elapsed_time()}] Warning: Could not filter by keyword: {e}")
        
        return active_features

def get_active_features(prompt: str, model_name: str = "meta-llama/Llama-2-7b-hf", 
                       sae_path: str = None, layer: int = 16, top_k: int = 10) -> List[Dict]:
    """
    Simple function to get active features for a prompt
    
    Args:
        prompt: Input text prompt
        model_name: HuggingFace model name
        sae_path: Path to SAE folder
        layer: SAE layer to analyze
        top_k: Number of top features to return
        
    Returns:
        List of dictionaries with feature info and activation values
    """
    checker = ConditionalChecker(model_name, sae_path)
    return checker.get_active_features(prompt, layer, top_k)

def check_condition(prompt: str, feature_id: int, model_name: str = "meta-llama/Llama-2-7b-hf",
                   sae_path: str = None, layer: int = 16, operator: str = "greater_than",
                   threshold: float = 0.0, use_percentage: bool = False) -> Dict:
    """
    Simple function to check a single condition
    
    Args:
        prompt: Input text prompt
        feature_id: Feature ID to check
        model_name: HuggingFace model name
        sae_path: Path to SAE folder
        layer: SAE layer to analyze
        operator: "greater_than" or "less_than"
        threshold: Threshold value to compare against
        use_percentage: Whether to use percentage (0-100) or raw activation
        
    Returns:
        Dictionary with condition result and activation info
    """
    checker = ConditionalChecker(model_name, sae_path)
    return checker.check_condition(prompt, feature_id, layer, operator, threshold, use_percentage)

def check_multiple_conditions(prompt: str, conditions: List[Dict], 
                             model_name: str = "meta-llama/Llama-2-7b-hf",
                             sae_path: str = None, layer: int = 16, 
                             logic_type: str = "AND") -> Dict:
    """
    Simple function to check multiple conditions with AND/OR logic
    
    Args:
        prompt: Input text prompt
        conditions: List of condition dictionaries
        model_name: HuggingFace model name
        sae_path: Path to SAE folder
        layer: SAE layer to analyze
        logic_type: "AND" or "OR" logic
        
    Returns:
        Dictionary with overall result and individual condition results
    """
    checker = ConditionalChecker(model_name, sae_path)
    return checker.check_multiple_conditions(prompt, conditions, layer, logic_type)

def search_features_by_prompt(prompt: str, model_name: str = "meta-llama/Llama-2-7b-hf",
                             sae_path: str = None, layer: int = 16, 
                             top_k: int = 10, keyword: str = None) -> List[Dict]:
    """
    Simple function to search for features by prompt
    
    Args:
        prompt: Input text prompt
        model_name: HuggingFace model name
        sae_path: Path to SAE folder
        layer: SAE layer to analyze
        top_k: Number of top features to return
        keyword: Optional keyword to filter features
        
    Returns:
        List of dictionaries with feature info and activation values
    """
    checker = ConditionalChecker(model_name, sae_path)
    return checker.search_features_by_prompt(prompt, layer, top_k, keyword)

def main():
    """Example usage"""
    print("=== CONDITIONAL CHECK MODULE ===")
    
    # Example 1: Get active features for a prompt
    print("\n1. Getting active features for a prompt:")
    prompt = "What are the side effects of aspirin?"
    active_features = get_active_features(prompt, top_k=5)
    
    print(f"Top 5 active features for: '{prompt}'")
    for i, feature in enumerate(active_features, 1):
        print(f"{i}. Feature {feature['feature_id']}: {feature['max_activation']:.3f} ({feature['activation_percentage']:.1f}%)")
    
    # Example 2: Check a single condition
    print("\n2. Checking a single condition:")
    if active_features:
        feature_id = active_features[0]['feature_id']
        result = check_condition(
            prompt=prompt,
            feature_id=feature_id,
            operator="greater_than",
            threshold=5.0,
            use_percentage=True
        )
        print(f"Feature {feature_id} > 5.0%: {result['condition_met']}")
        print(f"Actual activation: {result['activation_percentage']:.1f}%")
    
    # Example 3: Check multiple conditions with AND logic
    print("\n3. Checking multiple conditions with AND logic:")
    if len(active_features) >= 2:
        conditions = [
            {
                'feature_id': active_features[0]['feature_id'],
                'operator': 'greater_than',
                'threshold': 5.0,
                'use_percentage': True
            },
            {
                'feature_id': active_features[1]['feature_id'],
                'operator': 'greater_than',
                'threshold': 3.0,
                'use_percentage': True
            }
        ]
        
        result = check_multiple_conditions(prompt, conditions, logic_type="AND")
        print(f"AND logic result: {result['overall_met']}")
        print(f"Met {result['met_conditions']}/{result['total_conditions']} conditions")
    
    # Example 4: Search features by prompt with keyword
    print("\n4. Searching features by prompt with keyword:")
    medical_features = search_features_by_prompt(
        prompt=prompt,
        keyword="medical",
        top_k=3
    )
    
    print(f"Medical-related features for: '{prompt}'")
    for i, feature in enumerate(medical_features, 1):
        label = feature.get('label', f'Feature {feature["feature_id"]}')
        print(f"{i}. {label}: {feature['max_activation']:.3f} (similarity: {feature.get('similarity', 'N/A')})")

if __name__ == "__main__":
    main()
