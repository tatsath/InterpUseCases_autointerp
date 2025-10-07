#!/usr/bin/env python3
"""
Simple SAE Latent Vector UMAP Visualization
Loads latent vectors from trained SAE models and creates clustering visualization
"""

import torch
import numpy as np
import umap
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from safetensors import safe_open
import os

def load_sae_vectors(model_path, layer_name="layers.19"):
    """Load latent vectors from SAE model."""
    print(f"Loading SAE vectors from {model_path}...")
    
    sae_path = os.path.join(model_path, layer_name, "sae.safetensors")
    
    with safe_open(sae_path, framework="pt", device="cpu") as f:
        # Load decoder weights (these represent the latent directions)
        decoder_weights = f.get_tensor("W_dec")  # Shape: [num_latents, d_in]
    
    print(f"Loaded {decoder_weights.shape[0]} latent vectors of dimension {decoder_weights.shape[1]}")
    return decoder_weights

def create_umap_clustering(vectors, n_clusters=15):
    """Create UMAP visualization with clustering."""
    print("Creating UMAP embedding...")
    
    # UMAP dimensionality reduction
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(vectors)
    
    # K-means clustering
    print(f"Creating {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(vectors)
    
    return embedding, cluster_labels

def visualize_clusters(embedding, cluster_labels, output_path):
    """Create UMAP visualization similar to the example."""
    print("Creating visualization...")
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Color palette
    colors = plt.cm.Set3(np.linspace(0, 1, len(set(cluster_labels))))
    
    # Plot points colored by cluster
    for i, cluster_id in enumerate(sorted(set(cluster_labels))):
        mask = cluster_labels == cluster_id
        ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                  c=[colors[i]], alpha=0.7, s=50, 
                  label=f'Cluster {cluster_id}')
    
    # Add cluster centers
    for cluster_id in sorted(set(cluster_labels)):
        mask = cluster_labels == cluster_id
        center_x = np.mean(embedding[mask, 0])
        center_y = np.mean(embedding[mask, 1])
        ax.scatter(center_x, center_y, c='black', s=200, marker='x', linewidth=3)
    
    ax.set_title('SAE Latent Vectors - UMAP Clustering', fontsize=16, fontweight='bold')
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {output_path}")

def main():
    # Model paths
    model_400 = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama3.1_8b_layer19_k32_latents400_wikitext103_optimized"
    model_800 = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama3.1_8b_layer19_k32_latents800_wikitext103_optimized"
    
    # Load vectors from both models
    vectors_400 = load_sae_vectors(model_400)
    vectors_800 = load_sae_vectors(model_800)
    
    # Combine vectors
    all_vectors = np.vstack([vectors_400, vectors_800])
    print(f"Total vectors: {all_vectors.shape[0]}")
    
    # Create UMAP and clustering
    embedding, cluster_labels = create_umap_clustering(all_vectors, n_clusters=15)
    
    # Visualize
    visualize_clusters(embedding, cluster_labels, './sae_latent_umap.png')
    
    # Save results
    np.save('./sae_embedding.npy', embedding)
    np.save('./sae_clusters.npy', cluster_labels)
    np.save('./sae_vectors.npy', all_vectors)
    
    print("Results saved:")
    print("- sae_latent_umap.png (visualization)")
    print("- sae_embedding.npy (UMAP coordinates)")
    print("- sae_clusters.npy (cluster assignments)")
    print("- sae_vectors.npy (original vectors)")

if __name__ == "__main__":
    main()
