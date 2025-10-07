#!/usr/bin/env python3
"""
SAE Label Clustering Script
Clusters SAE labels using TF-IDF vectorization, cosine similarity, and K-means clustering.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import argparse
import os
import re
from sklearn.feature_extraction.text import CountVectorizer

def load_data(csv_path):
    """Load the SAE results CSV file."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records")
    return df

def preprocess_labels(labels):
    """Preprocess labels for better clustering."""
    # Remove common financial terms that might not be discriminative
    stop_words = ['financial', 'business', 'market', 'context', 'contexts', 'terms', 'actions', 'metrics']
    
    processed_labels = []
    for label in labels:
        # Convert to lowercase and remove punctuation
        label = str(label).lower()
        # Remove stop words
        for word in stop_words:
            label = label.replace(word, '')
        # Clean up extra spaces
        label = ' '.join(label.split())
        processed_labels.append(label)
    
    return processed_labels

def compute_cosine_similarity_matrix(labels):
    """Compute cosine similarity matrix using TF-IDF vectors."""
    print("Computing TF-IDF vectors...")
    
    # Use TF-IDF to vectorize the labels
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    
    tfidf_matrix = vectorizer.fit_transform(labels)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # Compute cosine similarity
    print("Computing cosine similarity matrix...")
    cosine_sim = cosine_similarity(tfidf_matrix)
    
    return cosine_sim, vectorizer, tfidf_matrix

def find_optimal_clusters(cosine_sim, min_clusters=100, max_clusters=200):
    """Find optimal number of clusters with minimum of 100 clusters."""
    print(f"Finding optimal number of clusters (minimum: {min_clusters})...")
    
    # Use cosine similarity as distance (1 - similarity)
    distance_matrix = 1 - cosine_sim
    
    # Calculate reasonable number of clusters based on data size
    n_samples = len(cosine_sim)
    # Aim for clusters with at least 10-20 samples each
    max_reasonable_clusters = min(max_clusters, n_samples // 10)
    min_reasonable_clusters = max(min_clusters, n_samples // 50)
    
    # Try different numbers of clusters
    inertias = []
    K_range = range(min_reasonable_clusters, max_reasonable_clusters + 1, 10)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(distance_matrix)
        inertias.append(kmeans.inertia_)
    
    # Find elbow point (simple heuristic)
    if len(inertias) > 1:
        # Calculate second derivative to find elbow
        second_deriv = np.diff(inertias, 2)
        if len(second_deriv) > 0:
            optimal_k = K_range[np.argmax(second_deriv) + 2]
        else:
            optimal_k = min_reasonable_clusters
    else:
        optimal_k = min_reasonable_clusters  # Default fallback
    
    print(f"Optimal number of clusters: {optimal_k}")
    return optimal_k, inertias, K_range

def perform_clustering(cosine_sim, n_clusters):
    """Perform K-means clustering on cosine similarity matrix."""
    print(f"Performing K-means clustering with {n_clusters} clusters...")
    
    # Convert similarity to distance
    distance_matrix = 1 - cosine_sim
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(distance_matrix)
    
    return cluster_labels, kmeans

def generate_cluster_names(df_clustered, vectorizer, tfidf_matrix, cluster_labels):
    """Generate meaningful names for clusters based on most frequent terms."""
    print("Generating cluster names...")
    
    cluster_names = {}
    
    for cluster_id in sorted(set(cluster_labels)):
        # Get all labels in this cluster
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
        cluster_labels_text = cluster_data['label'].tolist()
        
        if len(cluster_labels_text) == 0:
            cluster_names[cluster_id] = f"Cluster_{cluster_id}"
            continue
        
        # Extract key terms from cluster labels
        all_text = ' '.join(cluster_labels_text).lower()
        
        # Remove common stop words and clean text
        stop_words = {'financial', 'business', 'market', 'context', 'contexts', 'terms', 'actions', 
                     'metrics', 'related', 'specific', 'focusing', 'especially', 'including', 
                     'various', 'different', 'types', 'forms', 'ways', 'methods', 'processes'}
        
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

def analyze_clusters(df, cluster_labels, vectorizer, tfidf_matrix):
    """Analyze and display cluster results."""
    print("\n" + "="*50)
    print("CLUSTER ANALYSIS RESULTS")
    print("="*50)
    
    # Add cluster labels to dataframe
    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels
    
    # Generate cluster names
    cluster_names = generate_cluster_names(df_clustered, vectorizer, tfidf_matrix, cluster_labels)
    
    # Add cluster names to dataframe
    df_clustered['cluster_name'] = df_clustered['cluster'].map(cluster_names)
    
    # Get cluster statistics
    cluster_counts = Counter(cluster_labels)
    print(f"\nTotal clusters: {len(cluster_counts)}")
    print(f"Cluster distribution (showing top 20 largest clusters):")
    
    # Sort clusters by size and show top 20
    sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
    for cluster_id, count in sorted_clusters[:20]:
        cluster_name = cluster_names[cluster_id]
        print(f"  {cluster_name} (ID: {cluster_id}): {count} features")
    
    if len(sorted_clusters) > 20:
        print(f"  ... and {len(sorted_clusters) - 20} more clusters")
    
    # Analyze top 10 clusters in detail
    print(f"\nDetailed analysis of top 10 clusters:")
    print("-" * 50)
    
    for cluster_id, count in sorted_clusters[:10]:
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
        cluster_name = cluster_names[cluster_id]
        
        print(f"\n{cluster_name} (ID: {cluster_id}, {len(cluster_data)} features):")
        print(f"  Layers: {sorted(cluster_data['layer'].unique())}")
        print(f"  Avg F1 Score: {cluster_data['f1_score'].mean():.3f}")
        print(f"  Avg Accuracy: {cluster_data['accuracy'].mean():.3f}")
        
        # Show top labels in this cluster
        print(f"  Sample labels:")
        for i, (_, row) in enumerate(cluster_data.head(2).iterrows()):
            print(f"    - {row['label']}")
        if len(cluster_data) > 2:
            print(f"    ... and {len(cluster_data) - 2} more")
    
    return df_clustered, cluster_names

def visualize_clusters(tfidf_matrix, cluster_labels, cluster_names, output_dir):
    """Create visualizations of the clusters."""
    print("Creating visualizations...")
    
    # Reduce dimensionality for visualization
    pca = PCA(n_components=2, random_state=42)
    tfidf_2d = pca.fit_transform(tfidf_matrix.toarray())
    
    # Create scatter plot with better handling for many clusters
    plt.figure(figsize=(15, 10))
    
    # Use a colormap that can handle many clusters
    n_clusters = len(set(cluster_labels))
    if n_clusters <= 20:
        cmap = 'tab20'
    else:
        cmap = 'viridis'  # Better for many clusters
    
    scatter = plt.scatter(tfidf_2d[:, 0], tfidf_2d[:, 1], c=cluster_labels, cmap=cmap, alpha=0.6, s=20)
    plt.colorbar(scatter, label='Cluster ID')
    plt.title(f'SAE Label Clusters (PCA Visualization) - {n_clusters} Clusters')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    
    # Save plot
    plot_path = os.path.join(output_dir, 'cluster_visualization.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {plot_path}")
    
    # Create a cluster size distribution plot
    cluster_counts = Counter(cluster_labels)
    plt.figure(figsize=(12, 6))
    
    # Sort clusters by size
    sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
    cluster_ids, counts = zip(*sorted_clusters)
    
    plt.bar(range(len(cluster_ids)), counts)
    plt.title('Cluster Size Distribution')
    plt.xlabel('Cluster Rank (by size)')
    plt.ylabel('Number of Features')
    plt.xticks(range(0, len(cluster_ids), max(1, len(cluster_ids)//20)))
    
    # Save cluster size plot
    size_plot_path = os.path.join(output_dir, 'cluster_size_distribution.png')
    plt.savefig(size_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Cluster size distribution saved to: {size_plot_path}")

def save_results(df_clustered, output_dir):
    """Save clustering results to CSV."""
    output_path = os.path.join(output_dir, 'clustered_sae_labels.csv')
    df_clustered.to_csv(output_path, index=False)
    print(f"Clustered results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Cluster SAE labels using cosine similarity and K-means')
    parser.add_argument('--input', '-i', 
                       default='../FinanceLabeling/all_layers_financial_results/results_summary_all_layers_large_model_yahoo_finance.csv',
                       help='Path to input CSV file')
    parser.add_argument('--clusters', '-k', type=int, default=None,
                       help='Number of clusters (auto-detect if not specified)')
    parser.add_argument('--min-clusters', type=int, default=100,
                       help='Minimum number of clusters (default: 100)')
    parser.add_argument('--output', '-o', default='.',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load data
    df = load_data(args.input)
    
    # Preprocess labels
    print("Preprocessing labels...")
    processed_labels = preprocess_labels(df['label'].values)
    
    # Compute cosine similarity
    cosine_sim, vectorizer, tfidf_matrix = compute_cosine_similarity_matrix(processed_labels)
    
    # Find optimal number of clusters
    if args.clusters is None:
        optimal_k, inertias, K_range = find_optimal_clusters(cosine_sim, min_clusters=args.min_clusters)
    else:
        optimal_k = args.clusters
        print(f"Using specified number of clusters: {optimal_k}")
    
    # Perform clustering
    cluster_labels, kmeans = perform_clustering(cosine_sim, optimal_k)
    
    # Analyze results
    df_clustered, cluster_names = analyze_clusters(df, cluster_labels, vectorizer, tfidf_matrix)
    
    # Create visualizations
    visualize_clusters(tfidf_matrix, cluster_labels, cluster_names, args.output)
    
    # Save results
    save_results(df_clustered, args.output)
    
    # Save cluster names mapping
    cluster_mapping_path = os.path.join(args.output, 'cluster_names_mapping.csv')
    cluster_mapping_df = pd.DataFrame([
        {'cluster_id': cluster_id, 'cluster_name': cluster_name} 
        for cluster_id, cluster_name in cluster_names.items()
    ])
    cluster_mapping_df.to_csv(cluster_mapping_path, index=False)
    print(f"Cluster names mapping saved to: {cluster_mapping_path}")
    
    print(f"\nClustering completed successfully!")
    print(f"Results saved in: {args.output}")

if __name__ == "__main__":
    main()
