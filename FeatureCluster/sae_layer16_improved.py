#!/usr/bin/env python3
"""
Improved SAE Layer 16 UMAP with Optimal Clustering
Uses elbow method to find optimal number of clusters and correct CSV file
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
import re
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot

def load_sae_vectors(model_path):
    """Load SAE decoder vectors from layer 16."""
    sae_path = os.path.join(model_path, "layers.16", "sae.safetensors")
    with safe_open(sae_path, framework="pt", device="cpu") as f:
        W_dec = f.get_tensor("W_dec")  # Shape: [num_latents, d_in]
    return W_dec.numpy()

def load_labels(csv_path, layer=16):
    """Load labels for layer 16 from the correct CSV."""
    df = pd.read_csv(csv_path)
    layer_df = df[df['layer'] == layer].reset_index(drop=True)
    return layer_df

def find_optimal_clusters(vectors, max_clusters=30):
    """Find optimal number of clusters using elbow method on cosine similarity."""
    print("Finding optimal number of clusters...")
    
    # Compute cosine similarity matrix
    cosine_sim = cosine_similarity(vectors)
    distance_matrix = 1 - cosine_sim
    
    # Try different numbers of clusters - increased range for more granular clustering
    inertias = []
    K_range = range(2, min(max_clusters + 1, len(vectors) // 3))  # More clusters allowed
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(distance_matrix)
        inertias.append(kmeans.inertia_)
    
    # Find elbow point using second derivative
    if len(inertias) > 2:
        second_deriv = np.diff(inertias, 2)
        optimal_k = K_range[np.argmax(second_deriv) + 2]
        # Ensure we have enough clusters for meaningful separation
        optimal_k = max(optimal_k, 15)  # Minimum 15 clusters
    else:
        optimal_k = 15  # Default fallback
    
    print(f"Optimal number of clusters: {optimal_k}")
    return optimal_k, inertias, K_range

def create_cosine_clustering(vectors, n_clusters):
    """Cluster vectors using cosine similarity."""
    # Compute cosine similarity matrix
    cosine_sim = cosine_similarity(vectors)
    distance_matrix = 1 - cosine_sim
    
    # K-means clustering on cosine distance
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(distance_matrix)
    
    return cluster_labels, cosine_sim

def generate_cluster_names(cluster_labels, labels_df):
    """Generate meaningful cluster names based on all labels in each cluster."""
    cluster_names = {}
    
    for cluster_id in sorted(set(cluster_labels)):
        # Get all labels in this cluster
        cluster_features = np.where(cluster_labels == cluster_id)[0]
        cluster_labels_text = []
        
        for feat_idx in cluster_features:
            if feat_idx < len(labels_df):
                cluster_labels_text.append(labels_df.iloc[feat_idx]['label'])
        
        if not cluster_labels_text:
            cluster_names[cluster_id] = f"Cluster_{cluster_id}"
            continue
        
        # Extract key terms from all labels in cluster
        all_text = ' '.join(cluster_labels_text).lower()
        
        # Remove common stop words
        stop_words = {'and', 'or', 'in', 'of', 'the', 'a', 'an', 'to', 'for', 'with', 'by', 
                     'from', 'at', 'on', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'can', 'must', 'shall', 'context', 'contexts',
                     'specific', 'often', 'including', 'related', 'various', 'different'}
        
        # Extract meaningful words (3+ characters, not stop words)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text)
        words = [w for w in words if w not in stop_words]
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Get top 3 most frequent words
        top_words = [word for word, count in word_counts.most_common(3)]
        
        # Create cluster name
        if top_words:
            if len(top_words) == 1:
                cluster_name = f"{top_words[0].title()}_Features"
            elif len(top_words) == 2:
                cluster_name = f"{top_words[0].title()}_{top_words[1].title()}_Features"
            else:
                cluster_name = f"{top_words[0].title()}_{top_words[1].title()}_{top_words[2].title()}_Features"
        else:
            cluster_name = f"Cluster_{cluster_id}"
        
        cluster_names[cluster_id] = cluster_name
    
    return cluster_names

def create_interactive_umap(vectors, cluster_labels, labels_df, cluster_names, output_path):
    """Create interactive UMAP visualization with hover labels."""
    # UMAP embedding with better parameters for clarity
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=10, min_dist=0.05, spread=1.5)
    embedding = reducer.fit_transform(vectors)
    
    # Prepare data for interactive plot
    plot_data = []
    for i, (x, y) in enumerate(embedding):
        cluster_id = cluster_labels[i]
        cluster_name = cluster_names[cluster_id]
        
        # Get label for this feature
        if i < len(labels_df):
            feature_label = labels_df.iloc[i]['label']
            feature_id = labels_df.iloc[i]['feature']
        else:
            feature_label = f"Feature {i}"
            feature_id = i
        
        plot_data.append({
            'x': x,
            'y': y,
            'cluster': cluster_id,
            'cluster_name': cluster_name,
            'feature_id': feature_id,
            'label': feature_label,
            'hover_text': f"Feature {feature_id}<br>Cluster: {cluster_name}<br>Label: {feature_label}"
        })
    
    # Create DataFrame for plotly
    df_plot = pd.DataFrame(plot_data)
    
    # Create interactive scatter plot
    fig = px.scatter(df_plot, 
                     x='x', y='y',
                     color='cluster_name',
                     hover_data=['feature_id', 'label'],
                     hover_name='cluster_name',
                     title='SAE Layer 16 - Interactive Clustering with Hover Labels',
                     labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'},
                     width=1200, height=800)
    
    # Update layout for better appearance
    fig.update_layout(
        title_font_size=16,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    # Update traces to show only the actual label name on hover
    fig.update_traces(
        marker=dict(size=8, opacity=0.7, line=dict(width=1, color='black')),
        hovertemplate='%{customdata[1]}<extra></extra>'  # Only show the actual label name
    )
    
    # Save as HTML file
    fig.write_html(output_path)
    
    return fig

def main():
    # Paths - using the correct CSV file
    model_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
    csv_path = "../FinanceLabeling/all_layers_financial_results/results_summary_all_layers_large_model.csv"
    
    print("Loading SAE vectors from layer 16...")
    vectors = load_sae_vectors(model_path)
    print(f"Loaded {vectors.shape[0]} vectors of dimension {vectors.shape[1]}")
    
    print("Loading labels for layer 16...")
    labels_df = load_labels(csv_path, layer=16)
    print(f"Loaded {len(labels_df)} labels for layer 16")
    
    # Find optimal number of clusters
    optimal_k, inertias, K_range = find_optimal_clusters(vectors)
    
    print("Creating cosine similarity clustering...")
    cluster_labels, cosine_sim = create_cosine_clustering(vectors, optimal_k)
    
    print("Generating cluster names...")
    cluster_names = generate_cluster_names(cluster_labels, labels_df)
    
    print("Creating interactive UMAP visualization...")
    create_interactive_umap(vectors, cluster_labels, labels_df, cluster_names, './layer16_optimal_umap.html')
    
    # Save results - only feature and label columns
    results_df = pd.DataFrame({
        'feature': labels_df['feature'].values,
        'label': labels_df['label'].values,
        'cluster': cluster_labels[:len(labels_df)],
        'cluster_name': [cluster_names[c] for c in cluster_labels[:len(labels_df)]]
    })
    results_df.to_csv('./layer16_optimal_clustered.csv', index=False)
    
    # Print cluster analysis
    print(f"\nCluster Analysis (Optimal k={optimal_k}):")
    print("="*50)
    for cluster_id in sorted(set(cluster_labels)):
        mask = cluster_labels == cluster_id
        count = np.sum(mask)
        cluster_name = cluster_names[cluster_id]
        print(f"{cluster_name}: {count} features")
    
    print("\nResults saved:")
    print("- layer16_optimal_umap.html (interactive visualization - hover to see labels)")
    print("- layer16_optimal_clustered.csv (clustered data)")

if __name__ == "__main__":
    main()
