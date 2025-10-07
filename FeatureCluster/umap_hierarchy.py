#!/usr/bin/env python3
"""
Hierarchical UMAP Visualization for Financial SAE Features
Creates semantic clustering with broader themes and subcategories
"""

import pandas as pd
import numpy as np
import umap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

def load_and_preprocess_data(csv_path):
    """Load data and preprocess labels."""
    print("Loading data...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} features")
    
    # Clean and preprocess labels
    labels = []
    for label in df['label'].values:
        # Convert to lowercase, remove punctuation, clean
        clean_label = re.sub(r'[^\w\s]', ' ', str(label).lower())
        clean_label = ' '.join(clean_label.split())
        labels.append(clean_label)
    
    return df, labels

def create_hierarchical_clusters(labels, n_main_clusters=8, n_sub_clusters=25):
    """Create hierarchical clustering structure."""
    print("Creating hierarchical clusters...")
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(labels)
    
    # Main category clustering
    main_kmeans = KMeans(n_clusters=n_main_clusters, random_state=42)
    main_clusters = main_kmeans.fit_predict(tfidf_matrix)
    
    # Subcategory clustering
    sub_kmeans = KMeans(n_clusters=n_sub_clusters, random_state=42)
    sub_clusters = sub_kmeans.fit_predict(tfidf_matrix)
    
    return tfidf_matrix, main_clusters, sub_clusters, vectorizer

def generate_theme_names(labels, clusters, vectorizer, level="main"):
    """Generate meaningful theme names for clusters."""
    theme_names = {}
    
    for cluster_id in sorted(set(clusters)):
        cluster_labels = [labels[i] for i, c in enumerate(clusters) if c == cluster_id]
        
        # Extract key terms
        all_text = ' '.join(cluster_labels).lower()
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text)
        
        # Remove common stop words
        stop_words = {'financial', 'business', 'market', 'context', 'terms', 'actions', 
                     'related', 'specific', 'focusing', 'especially', 'including'}
        words = [w for w in words if w not in stop_words]
        
        # Get most frequent words
        word_counts = Counter(words)
        top_words = [word for word, count in word_counts.most_common(3)]
        
        # Create theme name
        if level == "main":
            if len(top_words) >= 2:
                theme_name = f"{top_words[0].title()} & {top_words[1].title()}"
            else:
                theme_name = f"{top_words[0].title() if top_words else 'General'}"
        else:  # subcategory
            if len(top_words) >= 2:
                theme_name = f"{top_words[0].title()} {top_words[1].title()}"
            else:
                theme_name = f"{top_words[0].title() if top_words else 'General'}"
        
        theme_names[cluster_id] = theme_name
    
    return theme_names

def create_umap_visualization(tfidf_matrix, main_clusters, sub_clusters, main_themes, sub_themes, output_path):
    """Create hierarchical UMAP visualization."""
    print("Creating UMAP visualization...")
    
    # UMAP dimensionality reduction
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(tfidf_matrix.toarray())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 16))
    
    # Color palette for main themes
    main_colors = plt.cm.Set3(np.linspace(0, 1, len(set(main_clusters))))
    sub_colors = plt.cm.tab20(np.linspace(0, 1, len(set(sub_clusters))))
    
    # Plot points colored by main themes
    for i, main_cluster in enumerate(set(main_clusters)):
        mask = main_clusters == main_cluster
        ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                  c=[main_colors[i]], alpha=0.6, s=30, 
                  label=main_themes[main_cluster])
    
    # Add cluster centers
    for main_cluster in set(main_clusters):
        mask = main_clusters == main_cluster
        center_x = np.mean(embedding[mask, 0])
        center_y = np.mean(embedding[mask, 1])
        ax.scatter(center_x, center_y, c='black', s=200, marker='x', linewidth=3)
    
    # Add theme labels
    for main_cluster in set(main_clusters):
        mask = main_clusters == main_cluster
        center_x = np.mean(embedding[mask, 0])
        center_y = np.mean(embedding[mask, 1])
        ax.annotate(main_themes[main_cluster], 
                   (center_x, center_y), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_title('Financial SAE Features - Hierarchical Semantic Clustering', fontsize=16, fontweight='bold')
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"UMAP visualization saved to: {output_path}")

def main():
    # Load data
    csv_path = '../FinanceLabeling/all_layers_financial_results/results_summary_all_layers_large_model_yahoo_finance.csv'
    df, labels = load_and_preprocess_data(csv_path)
    
    # Create hierarchical clusters
    tfidf_matrix, main_clusters, sub_clusters, vectorizer = create_hierarchical_clusters(labels)
    
    # Generate theme names
    main_themes = generate_theme_names(labels, main_clusters, vectorizer, "main")
    sub_themes = generate_theme_names(labels, sub_clusters, vectorizer, "sub")
    
    # Print theme hierarchy
    print("\n" + "="*60)
    print("HIERARCHICAL THEME STRUCTURE")
    print("="*60)
    
    for main_cluster in sorted(set(main_clusters)):
        print(f"\n📊 {main_themes[main_cluster]}")
        main_mask = main_clusters == main_cluster
        sub_clusters_in_main = sub_clusters[main_mask]
        
        for sub_cluster in sorted(set(sub_clusters_in_main)):
            sub_mask = (main_clusters == main_cluster) & (sub_clusters == sub_cluster)
            count = np.sum(sub_mask)
            print(f"  ├── {sub_themes[sub_cluster]} ({count} features)")
    
    # Create visualization
    create_umap_visualization(tfidf_matrix, main_clusters, sub_clusters, 
                            main_themes, sub_themes, './hierarchical_umap.png')
    
    # Save results
    df['main_theme'] = [main_themes[c] for c in main_clusters]
    df['sub_theme'] = [sub_themes[c] for c in sub_clusters]
    df.to_csv('./hierarchical_themes.csv', index=False)
    
    print(f"\nResults saved to: ./hierarchical_themes.csv")
    print("UMAP visualization completed!")

if __name__ == "__main__":
    main()
