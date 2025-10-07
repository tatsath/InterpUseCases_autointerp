#!/usr/bin/env python3
"""
Standalone script to generate HTML exports of UMAP visualizations
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.colors as pc
import colorsys
from safetensors import safe_open
import umap
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def load_sae_vectors():
    """Load SAE vectors for layer 16."""
    model_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
    sae_path = os.path.join(model_path, "layers.16", "sae.safetensors")
    
    with safe_open(sae_path, framework="pt", device="cpu") as f:
        vectors = f.get_tensor("W_dec").numpy()
    
    return vectors

def load_general_labels():
    """Load general feature labels."""
    csv_path = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/FinanceLabeling/multi_layer_full_results/results_summary_layer16.csv"
    return pd.read_csv(csv_path)

def load_financial_labels():
    """Load financial feature labels."""
    csv_path = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/FinanceLabeling/all_layers_financial_results/results_summary_all_layers_large_model_yahoo_finance.csv"
    df = pd.read_csv(csv_path)
    return df[df['layer'] == 16].reset_index(drop=True)

def create_clustering(vectors, n_clusters=50):
    """Create clustering of vectors based on cosine similarity."""
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(vectors)
    
    # Convert to distance matrix (1 - similarity)
    distance_matrix = 1 - similarity_matrix
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(distance_matrix)
    
    return cluster_labels

def create_umap_embedding(vectors):
    """Create UMAP embedding."""
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=10, min_dist=0.05, spread=1.5)
    embedding = reducer.fit_transform(vectors)
    return embedding

def create_visualization(vectors, labels_df, n_clusters, title):
    """Create UMAP visualization with clustering."""
    # Create clustering
    cluster_labels = create_clustering(vectors, n_clusters)
    
    # Create UMAP embedding
    embedding = create_umap_embedding(vectors)
    
    # Create a mapping from feature numbers to labels
    feature_to_label = {}
    for _, row in labels_df.iterrows():
        feature_to_label[row['feature']] = row['label']
    
    # Prepare data for visualization
    plot_data = []
    for i, (x, y) in enumerate(embedding):
        cluster_id = cluster_labels[i]
        
        if i in feature_to_label:
            feature_label = feature_to_label[i]
            feature_id = i
        else:
            feature_label = f"Feature {i} (No label)"
            feature_id = i
        
        plot_data.append({
            'x': x,
            'y': y,
            'cluster': cluster_id,
            'feature_id': feature_id,
            'label': feature_label
        })
    
    df_plot = pd.DataFrame(plot_data)
    
    # Create color palette
    n_clusters = len(set(cluster_labels))
    base_colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F',
        '#BB8FCE', '#85C1E9', '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2', '#A9DFBF',
        '#F9E79F', '#AED6F1', '#D5DBDB', '#FADBD8', '#D1F2EB', '#FCF3CF', '#E8DAEF', '#D5F4E6',
        '#FFF2CC', '#E1F5FE', '#F3E5F5', '#E8F5E8', '#FFF8E1', '#E3F2FD', '#FCE4EC', '#F1F8E9',
        '#FFEBEE', '#E0F2F1', '#FFF3E0', '#E8EAF6', '#F9FBE7', '#EFEBE9', '#FAFAFA', '#ECEFF1'
    ]
    
    # Generate additional colors if needed
    if n_clusters > len(base_colors):
        additional_colors = []
        for i in range(n_clusters - len(base_colors)):
            hue = (i * 137.5) % 360
            saturation = 0.7 + (i % 3) * 0.1
            value = 0.8 + (i % 2) * 0.2
            rgb = colorsys.hsv_to_rgb(hue/360, saturation, value)
            rgb_int = [int(c * 255) for c in rgb]
            additional_colors.append(f'rgb({rgb_int[0]}, {rgb_int[1]}, {rgb_int[2]})')
        base_colors.extend(additional_colors)
    
    cluster_colors = base_colors[:n_clusters]
    
    # Ensure cluster column is treated as categorical
    df_plot['cluster'] = df_plot['cluster'].astype(str)
    
    # Create interactive plot
    fig = px.scatter(df_plot, 
                     x='x', y='y',
                     color='cluster',
                     hover_data=['feature_id', 'label'],
                     title=title,
                     labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'},
                     width=1000, height=700,
                     color_discrete_sequence=cluster_colors)
    
    # Update layout
    fig.update_layout(
        title_font_size=20,
        title_x=0.5,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=11, color='black'),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=2,
            itemsizing='constant',
            itemwidth=30
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12, color='black'),
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=1,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=1,
            zeroline=False
        )
    )
    
    # Update traces
    fig.update_traces(
        marker=dict(size=12, opacity=0.9, line=dict(width=2, color='black')),
        hovertemplate='<b>Feature %{customdata[0]}</b><br>%{customdata[1]}<extra></extra>',
        selector=dict(type='scatter')
    )
    
    return fig, cluster_labels, feature_to_label

def save_html_export(fig, title, filename):
    """Save the plotly figure as an HTML file."""
    html_content = fig.to_html(include_plotlyjs=True, full_html=True)
    
    # Add custom styling and title
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                text-align: center;
                margin-bottom: 20px;
            }}
            .info {{
                background-color: #e3f2fd;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
                border-left: 4px solid #2196f3;
            }}
            .stats {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-top: 20px;
                border-left: 4px solid #28a745;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{title}</h1>
            <div class="info">
                <strong>Interactive UMAP Visualization:</strong> Hover over points to see feature labels and cluster information. 
                Each cluster is assigned a distinct color for easy identification. This visualization shows how SAE features 
                cluster based on their vector directions (cosine similarity).
            </div>
            {html_content}
            <div class="stats">
                <h3>About This Visualization</h3>
                <p><strong>Clustering Method:</strong> K-means clustering based on cosine similarity between SAE decoder vectors</p>
                <p><strong>Dimensionality Reduction:</strong> UMAP (Uniform Manifold Approximation and Projection)</p>
                <p><strong>Features:</strong> SAE Layer 16 decoder vectors (400 features)</p>
                <p><strong>Interactive Features:</strong> Hover to see feature labels, zoom, pan, and toggle clusters</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(styled_html)
    
    return filename

def main():
    """Generate HTML exports for both general and financial features."""
    print("🧠 Generating HTML exports for SAE Layer 16 UMAP visualizations...")
    
    # Load SAE vectors
    print("Loading SAE vectors...")
    vectors = load_sae_vectors()
    
    # Generate General Features HTML
    print("Generating General Features visualization...")
    labels_df = load_general_labels()
    fig, cluster_labels, feature_to_label = create_visualization(
        vectors, labels_df, 50, "SAE Layer 16 - General Features Clustering"
    )
    filename1 = save_html_export(fig, "SAE Layer 16 - General Features Clustering", "general_features_umap.html")
    print(f"✅ General Features HTML saved: {filename1}")
    
    # Generate Financial Features HTML
    print("Generating Financial Features visualization...")
    labels_df = load_financial_labels()
    fig, cluster_labels, feature_to_label = create_visualization(
        vectors, labels_df, 50, "SAE Layer 16 - Financial Features Clustering"
    )
    filename2 = save_html_export(fig, "SAE Layer 16 - Financial Features Clustering", "financial_features_umap.html")
    print(f"✅ Financial Features HTML saved: {filename2}")
    
    print("\n🎉 HTML exports generated successfully!")
    print("You can now share these HTML files with others.")
    print("They contain fully interactive visualizations that work in any web browser.")

if __name__ == "__main__":
    main()
