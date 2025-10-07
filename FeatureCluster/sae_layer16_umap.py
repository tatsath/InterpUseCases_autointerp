#!/usr/bin/env python3
"""
SAE Layer 16 UMAP with Labels
Loads SAE vectors from layer 16, clusters using cosine similarity, and labels with CSV data
"""

import torch
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from safetensors import safe_open
import os

def load_sae_vectors(model_path):
    """Load SAE decoder vectors from layer 16."""
    sae_path = os.path.join(model_path, "layers.16", "sae.safetensors")
    with safe_open(sae_path, framework="pt", device="cpu") as f:
        W_dec = f.get_tensor("W_dec")  # Shape: [num_latents, d_in]
    return W_dec.numpy()

def load_labels(csv_path, layer=16):
    """Load labels for layer 16 from CSV."""
    df = pd.read_csv(csv_path)
    layer_df = df[df['layer'] == layer].reset_index(drop=True)
    return layer_df

def create_cosine_clustering(vectors, n_clusters=12):
    """Cluster vectors using cosine similarity."""
    # Compute cosine similarity matrix
    cosine_sim = cosine_similarity(vectors)
    distance_matrix = 1 - cosine_sim
    
    # K-means clustering on cosine distance
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(distance_matrix)
    
    return cluster_labels, cosine_sim

def create_umap_visualization(vectors, cluster_labels, labels_df, output_path):
    """Create UMAP visualization with cluster labels."""
    # UMAP embedding
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(vectors)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Color palette
    colors = plt.cm.Set3(np.linspace(0, 1, len(set(cluster_labels))))
    
    # Plot points colored by cluster
    for i, cluster_id in enumerate(sorted(set(cluster_labels))):
        mask = cluster_labels == cluster_id
        ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                  c=[colors[i]], alpha=0.7, s=60, 
                  label=f'Cluster {cluster_id}')
    
    # Add cluster centers with labels
    for cluster_id in sorted(set(cluster_labels)):
        mask = cluster_labels == cluster_id
        center_x = np.mean(embedding[mask, 0])
        center_y = np.mean(embedding[mask, 1])
        
        # Get sample labels for this cluster
        cluster_features = np.where(mask)[0]
        if len(cluster_features) > 0:
            # Get corresponding labels from CSV
            sample_features = cluster_features[:3]  # Take first 3 features
            sample_labels = []
            for feat_idx in sample_features:
                if feat_idx < len(labels_df):
                    label = labels_df.iloc[feat_idx]['label']
                    # Truncate long labels
                    if len(label) > 50:
                        label = label[:47] + "..."
                    sample_labels.append(label)
            
            # Create cluster label
            cluster_label = f"Cluster {cluster_id}\n" + "\n".join(sample_labels)
            
            ax.annotate(cluster_label, 
                       (center_x, center_y), 
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Mark cluster center
        ax.scatter(center_x, center_y, c='black', s=200, marker='x', linewidth=3)
    
    ax.set_title('SAE Layer 16 - Latent Vector Clustering with Labels', fontsize=16, fontweight='bold')
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Paths
    model_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
    csv_path = "../FinanceLabeling/all_layers_financial_results/results_summary_all_layers_large_model_yahoo_finance.csv"
    
    print("Loading SAE vectors from layer 16...")
    vectors = load_sae_vectors(model_path)
    print(f"Loaded {vectors.shape[0]} vectors of dimension {vectors.shape[1]}")
    
    print("Loading labels for layer 16...")
    labels_df = load_labels(csv_path, layer=16)
    print(f"Loaded {len(labels_df)} labels for layer 16")
    
    print("Creating cosine similarity clustering...")
    cluster_labels, cosine_sim = create_cosine_clustering(vectors, n_clusters=12)
    
    print("Creating UMAP visualization...")
    create_umap_visualization(vectors, cluster_labels, labels_df, './layer16_sae_umap.png')
    
    # Save results
    results_df = labels_df.copy()
    results_df['cluster'] = cluster_labels[:len(labels_df)]
    results_df.to_csv('./layer16_clustered_results.csv', index=False)
    
    print("Results saved:")
    print("- layer16_sae_umap.png (visualization)")
    print("- layer16_clustered_results.csv (clustered data)")

if __name__ == "__main__":
    main()
