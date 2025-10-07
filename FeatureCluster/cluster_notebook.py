#!/usr/bin/env python3
"""
SAE Label Clustering - Jupyter/IDE Version
Run this cell by cell or as a complete script
"""

# Cell 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

# Cell 2: Load and check data
csv_path = '/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/FinanceLabeling/all_layers_financial_results/results_summary_all_layers_large_model_yahoo_finance.csv'

print(f"Loading data from {csv_path}...")
print(f"File exists: {os.path.exists(csv_path)}")

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} records")
print(f"Columns: {df.columns.tolist()}")
print(f"Sample data:")
print(df.head())

# Cell 3: Preprocess labels
print("\nPreprocessing labels...")
labels = df['label'].values
processed_labels = [str(label).lower().strip() for label in labels]
print(f"Sample processed labels:")
for i in range(5):
    print(f"  {i+1}. {processed_labels[i]}")

# Cell 4: Vectorize using TF-IDF
print("\nComputing TF-IDF vectors...")
vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8
)

tfidf_matrix = vectorizer.fit_transform(processed_labels)
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

# Cell 5: Compute cosine similarity and cluster
print("\nComputing cosine similarity...")
cosine_sim = cosine_similarity(tfidf_matrix)
distance_matrix = 1 - cosine_sim

print("Performing K-means clustering...")
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(distance_matrix)

# Cell 6: Analyze results
df['cluster'] = cluster_labels

print(f"\nCluster Analysis:")
print("=" * 50)

for cluster_id in range(n_clusters):
    cluster_data = df[df['cluster'] == cluster_id]
    if len(cluster_data) > 0:
        print(f"\nCluster {cluster_id} ({len(cluster_data)} features):")
        print(f"  Layers: {sorted(cluster_data['layer'].unique())}")
        print(f"  Avg F1 Score: {cluster_data['f1_score'].mean():.3f}")
        print(f"  Avg Accuracy: {cluster_data['accuracy'].mean():.3f}")
        print(f"  Sample labels:")
        for i, (_, row) in enumerate(cluster_data.head(3).iterrows()):
            print(f"    - {row['label']}")

# Cell 7: Create visualization
print("\nCreating visualization...")
pca = PCA(n_components=2, random_state=42)
tfidf_2d = pca.fit_transform(tfidf_matrix.toarray())

plt.figure(figsize=(12, 8))
scatter = plt.scatter(tfidf_2d[:, 0], tfidf_2d[:, 1], c=cluster_labels, cmap='tab20', alpha=0.7)
plt.colorbar(scatter)
plt.title('SAE Label Clusters (PCA Visualization)')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.savefig('cluster_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# Cell 8: Save results
output_path = 'clustered_sae_labels.csv'
df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")
print("Clustering completed successfully!")

