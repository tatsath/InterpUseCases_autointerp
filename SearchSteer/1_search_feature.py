#!/usr/bin/env python3
"""
Semantic Feature Search Module

This module handles semantic search for features from SAE files.
It can automatically discover SAE files and search for relevant features.
"""

import pandas as pd
import numpy as np
import os
import time
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer
from safetensors import safe_open
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity

class SemanticFeatureSearch:
    def __init__(self, sae_path: str, layer: Optional[int] = None):
        """
        Initialize semantic feature search
        
        Args:
            sae_path: Path to SAE folder or specific SAE name
            layer: Specific layer to search (if None, auto-detect)
        """
        self.sae_path = sae_path
        self.layer = layer
        self.start_time = time.time()
        print(f"[{self._get_elapsed_time()}] Initializing SemanticFeatureSearch...")
        print(f"[{self._get_elapsed_time()}] Loading sentence transformer model...")
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"[{self._get_elapsed_time()}] Sentence transformer loaded successfully")
        self.features_df = None
        
        # Auto-detect SAE path and layer
        self._setup_sae_path()
        self._discover_features()
    
    def _get_elapsed_time(self):
        """Get formatted elapsed time"""
        elapsed = time.time() - self.start_time
        return f"{elapsed:06.1f}s"
    
    def _setup_sae_path(self):
        """Setup SAE path - handle both folder paths and SAE names"""
        print(f"[{self._get_elapsed_time()}] Setting up SAE path...")
        print(f"[{self._get_elapsed_time()}] Original SAE path: {self.sae_path}")
        
        # SAE name mapping for common cases
        sae_mapping = {
            "llama2_7b_hf": "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun",
            "llama2_7b_finance": "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_finance_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
        }
        
        # Check if it's a mapped name
        if self.sae_path in sae_mapping:
            self.sae_path = sae_mapping[self.sae_path]
            print(f"[{self._get_elapsed_time()}] Mapped to: {self.sae_path}")
        
        # Check if it's a direct folder path
        if not os.path.exists(self.sae_path):
            print(f"[{self._get_elapsed_time()}] Path not found, trying base path...")
            # Try to find SAE folder
            base_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/"
            potential_path = os.path.join(base_path, self.sae_path)
            if os.path.exists(potential_path):
                self.sae_path = potential_path
                print(f"[{self._get_elapsed_time()}] Found at: {self.sae_path}")
            else:
                raise FileNotFoundError(f"SAE path not found: {self.sae_path}")
        
        print(f"[{self._get_elapsed_time()}] Final SAE path: {self.sae_path}")
    
    def _discover_features(self):
        """Discover features from SAE files"""
        print(f"[{self._get_elapsed_time()}] Discovering features from SAE path: {self.sae_path}")
        
        # Find available layers
        print(f"[{self._get_elapsed_time()}] Scanning for available layers...")
        available_layers = []
        for layer in [4, 10, 16, 22, 28]:
            layer_path = os.path.join(self.sae_path, f"layers.{layer}")
            sae_path = os.path.join(layer_path, "sae.safetensors")
            print(f"[{self._get_elapsed_time()}] Checking layer {layer}: {sae_path}")
            if os.path.exists(sae_path):
                available_layers.append(layer)
                print(f"[{self._get_elapsed_time()}] ✓ Layer {layer} found")
            else:
                print(f"[{self._get_elapsed_time()}] ✗ Layer {layer} not found")
        
        if not available_layers:
            raise FileNotFoundError(f"No SAE files found in {self.sae_path}")
        
        print(f"[{self._get_elapsed_time()}] Found SAE files for layers: {available_layers}")
        
        # Use specified layer or default to 16
        target_layer = self.layer if self.layer in available_layers else (16 if 16 in available_layers else available_layers[0])
        print(f"[{self._get_elapsed_time()}] Using layer {target_layer}")
        
        self._load_features_for_layer(target_layer)
    
    def _load_features_for_layer(self, layer: int):
        """Load features for a specific layer"""
        layer_path = os.path.join(self.sae_path, f"layers.{layer}")
        sae_path = os.path.join(layer_path, "sae.safetensors")
        
        print(f"[{self._get_elapsed_time()}] Loading features for layer {layer}")
        print(f"[{self._get_elapsed_time()}] SAE file path: {sae_path}")
        
        if not os.path.exists(sae_path):
            raise FileNotFoundError(f"SAE file not found: {sae_path}")
        
        try:
            print(f"[{self._get_elapsed_time()}] Opening SAE file...")
            with safe_open(sae_path, framework="pt", device="cpu") as f:
                print(f"[{self._get_elapsed_time()}] Loading decoder weights...")
                decoder = f.get_tensor("W_dec")
                num_features = decoder.shape[0]
                print(f"[{self._get_elapsed_time()}] Found {num_features} features in layer {layer}")
                
                # Create feature data
                print(f"[{self._get_elapsed_time()}] Creating feature data...")
                features_data = []
                for i in range(num_features):
                    features_data.append({
                        'layer': layer,
                        'feature': i,
                        'label': f"Feature {i} (Layer {layer})",
                        'f1_score': 0.0
                    })
                
                self.features_df = pd.DataFrame(features_data)
                print(f"[{self._get_elapsed_time()}] Created {len(self.features_df)} feature entries for layer {layer}")
                
        except Exception as e:
            raise RuntimeError(f"Error loading SAE features: {e}")
    
    def search_features(self, keyword: str, top_k: int = 5, save_results: bool = True, output_file: str = None) -> List[Dict]:
        """Semantic search for features based on keyword"""
        if self.features_df is None:
            raise ValueError("Features not loaded")
        
        print(f"[{self._get_elapsed_time()}] Searching for '{keyword}' in {len(self.features_df)} features...")
        
        # Get embeddings for all feature labels
        print(f"[{self._get_elapsed_time()}] Computing embeddings for {len(self.features_df)} feature labels...")
        feature_labels = self.features_df['label'].tolist()
        feature_embeddings = self.semantic_model.encode(feature_labels)
        print(f"[{self._get_elapsed_time()}] Feature embeddings computed: {feature_embeddings.shape}")
        
        # Get embedding for search keyword
        print(f"[{self._get_elapsed_time()}] Computing embedding for keyword '{keyword}'...")
        keyword_embedding = self.semantic_model.encode([keyword])
        print(f"[{self._get_elapsed_time()}] Keyword embedding computed: {keyword_embedding.shape}")
        
        # Calculate cosine similarities
        print(f"[{self._get_elapsed_time()}] Calculating cosine similarities...")
        similarities = cosine_similarity(keyword_embedding, feature_embeddings)[0]
        print(f"[{self._get_elapsed_time()}] Similarities computed: {len(similarities)} values")
        
        # Get top-k most similar features
        print(f"[{self._get_elapsed_time()}] Finding top-{top_k} most similar features...")
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
        
        print(f"[{self._get_elapsed_time()}] Search completed. Found {len(results)} results.")
        
        # Save results to CSV if requested
        if save_results:
            if output_file is None:
                # Use standard filename format
                output_file = "1_feature_search_results.csv"
            
            print(f"[{self._get_elapsed_time()}] Saving results to {output_file}...")
            
            # Create DataFrame for saving
            save_data = []
            for result in results:
                save_data.append({
                    'feature_id': result['feature_id'],
                    'feature_name': result['label'],
                    'similarity': result['similarity'],
                    'layer': result['layer'],
                    'f1_score': result['f1_score'],
                    'search_keyword': keyword,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            
            results_df = pd.DataFrame(save_data)
            results_df.to_csv(output_file, index=False)
            print(f"[{self._get_elapsed_time()}] Results saved to {output_file}")
        
        return results
    
    def get_feature_info(self, feature_id: int) -> Dict:
        """Get information about a specific feature"""
        if self.features_df is None:
            raise ValueError("Features not loaded")
        
        feature_row = self.features_df[self.features_df['feature'] == feature_id]
        if feature_row.empty:
            raise ValueError(f"Feature {feature_id} not found")
        
        return {
            'feature_id': feature_id,
            'layer': feature_row.iloc[0]['layer'],
            'label': feature_row.iloc[0]['label'],
            'f1_score': feature_row.iloc[0]['f1_score']
        }
    
    def search_features_with_real_labels(self, keyword: str, top_k: int = 10, 
                                       csv_path: str = None) -> List[Dict]:
        """
        Search for features using real labels from CSV file (aligned with search_steer_app.py)
        
        Args:
            keyword: Search keyword
            top_k: Number of top results to return
            csv_path: Path to CSV file with real feature labels
            
        Returns:
            List of search results with real labels
        """
        if csv_path is None:
            csv_path = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/FinanceLabeling/multi_layer_full_results/multi_layer_full_layer16/results_summary.csv"
        
        print(f"[{self._get_elapsed_time()}] Searching with real labels for '{keyword}'...")
        
        try:
            # Load real feature labels
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Feature labels file not found: {csv_path}")
            
            df = pd.read_csv(csv_path)
            feature_labels = {}
            for _, row in df.iterrows():
                if row['layer'] == 16:  # Only layer 16 features
                    feature_labels[row['feature']] = row['label']
            
            if not feature_labels:
                print(f"[{self._get_elapsed_time()}] No feature labels loaded.")
                return []
            
            print(f"[{self._get_elapsed_time()}] Loaded {len(feature_labels)} real feature labels")
            
            # Get all feature labels and IDs
            feature_ids = list(feature_labels.keys())
            labels = list(feature_labels.values())
            
            # Compute embeddings for all labels
            print(f"[{self._get_elapsed_time()}] Computing embeddings for {len(labels)} real labels...")
            label_embeddings = self.semantic_model.encode(labels)
            
            # Compute embedding for search keyword
            print(f"[{self._get_elapsed_time()}] Computing embedding for keyword '{keyword}'...")
            keyword_embedding = self.semantic_model.encode([keyword])
            
            # Calculate similarities
            print(f"[{self._get_elapsed_time()}] Calculating cosine similarities...")
            similarities = cosine_similarity(keyword_embedding, label_embeddings)[0]
            
            # Get top-k most similar features
            print(f"[{self._get_elapsed_time()}] Finding top-{top_k} most similar features...")
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                feature_id = feature_ids[idx]
                label = labels[idx]
                similarity = similarities[idx]
                
                results.append({
                    'feature_id': feature_id,
                    'layer': 16,  # All features are from layer 16
                    'label': label,
                    'similarity': similarity,
                    'f1_score': 0.0  # Not available in this search
                })
            
            print(f"[{self._get_elapsed_time()}] Search completed. Found {len(results)} results.")
            return results
            
        except Exception as e:
            print(f"[{self._get_elapsed_time()}] Real label search failed: {e}")
            return []
    
    @staticmethod
    def load_search_results(csv_file: str) -> pd.DataFrame:
        """Load previously saved search results from CSV file"""
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Results file not found: {csv_file}")
        
        print(f"Loading search results from {csv_file}...")
        results_df = pd.read_csv(csv_file)
        print(f"Loaded {len(results_df)} results from {csv_file}")
        return results_df

def main():
    """Example usage"""
    print("=== SEMANTIC FEATURE SEARCH ===")
    
    # Example 1: Using SAE name with real labels (recommended)
    print("\n1. Using SAE name with real labels (recommended):")
    searcher = SemanticFeatureSearch("llama2_7b_hf", layer=16)
    results = searcher.search_features_with_real_labels("medical", top_k=5)
    
    print(f"Found {len(results)} features for 'medical':")
    for i, result in enumerate(results, 1):
        print(f"{i}. Feature {result['feature_id']} (Layer {result['layer']})")
        print(f"   Label: {result['label']}")
        print(f"   Similarity: {result['similarity']:.3f}")
        print()
    
    # Example 2: Using direct path with standard filename
    print("\n2. Using direct SAE path (with standard CSV filename):")
    searcher2 = SemanticFeatureSearch("/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun", layer=16)
    results2 = searcher2.search_features("financial performance", top_k=2, save_results=True)
    
    print(f"Found {len(results2)} features for 'financial performance':")
    for i, result in enumerate(results2, 1):
        print(f"{i}. Feature {result['feature_id']} (Layer {result['layer']})")
        print(f"   Label: {result['label']}")
        print(f"   Similarity: {result['similarity']:.3f}")
    
    # Example 3: Load and display saved results
    print("\n3. Loading saved results:")
    try:
        # Try to load the standard filename
        saved_results = SemanticFeatureSearch.load_search_results("1_feature_search_results.csv")
        print("Saved results columns:", saved_results.columns.tolist())
        print("Sample saved data:")
        print(saved_results.head())
    except FileNotFoundError:
        print("No saved results found to load")

if __name__ == "__main__":
    main()
