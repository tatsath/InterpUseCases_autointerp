# SAE Label Clustering

This script clusters SAE (Sparse Autoencoder) labels using cosine similarity and K-means clustering.

## Features

- **TF-IDF Vectorization**: Converts text labels to numerical vectors
- **Cosine Similarity**: Computes similarity between label vectors
- **K-means Clustering**: Groups similar labels into clusters
- **Automatic Cluster Detection**: Finds optimal number of clusters using elbow method
- **Visualization**: Creates PCA-based scatter plot of clusters
- **Detailed Analysis**: Provides cluster statistics and sample labels

## Usage

### Basic Usage
```bash
python cluster_sae_labels.py
```

### Custom Parameters
```bash
# Specify number of clusters
python cluster_sae_labels.py --clusters 10

# Use custom input file
python cluster_sae_labels.py --input /path/to/your/data.csv

# Specify output directory
python cluster_sae_labels.py --output /path/to/results/
```

## Output Files

- `clustered_sae_labels.csv`: Original data with cluster assignments
- `cluster_visualization.png`: PCA visualization of clusters

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Algorithm

1. **Preprocessing**: Clean and normalize label text
2. **Vectorization**: Convert labels to TF-IDF vectors
3. **Similarity**: Compute cosine similarity matrix
4. **Clustering**: Apply K-means clustering
5. **Analysis**: Generate cluster statistics and visualizations
