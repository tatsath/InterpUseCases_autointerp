#!/usr/bin/env python3
"""
Streamlit UMAP Viewer
Simple Streamlit app to display the interactive UMAP visualization
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import umap
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from safetensors import safe_open
import os
import colorsys

# Page config
st.set_page_config(
    page_title="SAE Layer 16 UMAP Visualization",
    page_icon="🧠",
    layout="wide"
)

@st.cache_data
def load_data():
    """Load SAE vectors and labels."""
    # Paths
    model_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
    csv_path = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/FinanceLabeling/multi_layer_full_results/results_summary_layer16.csv"
    
    # Load SAE vectors
    sae_path = os.path.join(model_path, "layers.16", "sae.safetensors")
    with safe_open(sae_path, framework="pt", device="cpu") as f:
        vectors = f.get_tensor("W_dec").numpy()
    
    # Load labels (this CSV already contains only layer 16 data)
    labels_df = pd.read_csv(csv_path)
    
    return vectors, labels_df

@st.cache_data
def load_financial_data():
    """Load SAE vectors and financial labels."""
    # Paths
    model_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
    csv_path = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/FinanceLabeling/all_layers_financial_results/results_summary_all_layers_large_model_yahoo_finance.csv"
    
    # Load SAE vectors
    sae_path = os.path.join(model_path, "layers.16", "sae.safetensors")
    with safe_open(sae_path, framework="pt", device="cpu") as f:
        vectors = f.get_tensor("W_dec").numpy()
    
    # Load financial labels for layer 16
    df = pd.read_csv(csv_path)
    labels_df = df[df['layer'] == 16].reset_index(drop=True)
    
    return vectors, labels_df

@st.cache_data
def create_clustering(vectors, n_clusters=50):
    """Create clustering of vectors."""
    # Compute cosine similarity matrix
    cosine_sim = cosine_similarity(vectors)
    distance_matrix = 1 - cosine_sim
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(distance_matrix)
    
    return cluster_labels

@st.cache_data
def create_umap_embedding(vectors):
    """Create UMAP embedding."""
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=10, min_dist=0.05, spread=1.5)
    embedding = reducer.fit_transform(vectors)
    return embedding

def create_visualization(vectors, labels_df, n_clusters, title):
    """Create UMAP visualization with clustering."""
    # Create clustering
    with st.spinner("Creating clusters..."):
        cluster_labels = create_clustering(vectors, n_clusters)
    
    # Create UMAP embedding
    with st.spinner("Creating UMAP embedding..."):
        embedding = create_umap_embedding(vectors)
    
    # Create a mapping from feature numbers to labels for layer 16
    feature_to_label = {}
    for _, row in labels_df.iterrows():
        feature_to_label[row['feature']] = row['label']
    
    # Prepare data for visualization
    plot_data = []
    for i, (x, y) in enumerate(embedding):
        cluster_id = cluster_labels[i]
        
        # Use the feature number as the index to get the correct label
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
    
    # Create a better color palette for clusters with truly different colors
    n_clusters = len(set(cluster_labels))
    
    # Use a combination of different color palettes for maximum variety
    base_colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F',
        '#BB8FCE', '#85C1E9', '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2', '#A9DFBF',
        '#F9E79F', '#AED6F1', '#D5DBDB', '#FADBD8', '#D1F2EB', '#FCF3CF', '#E8DAEF', '#D5F4E6',
        '#FFF2CC', '#E1F5FE', '#F3E5F5', '#E8F5E8', '#FFF8E1', '#E3F2FD', '#FCE4EC', '#F1F8E9',
        '#FFEBEE', '#E0F2F1', '#FFF3E0', '#E8EAF6', '#F9FBE7', '#EFEBE9', '#FAFAFA', '#ECEFF1'
    ]
    
    # If we need more colors, generate additional distinct colors
    if n_clusters > len(base_colors):
        additional_colors = []
        for i in range(n_clusters - len(base_colors)):
            # Generate distinct colors using simple RGB combinations
            hue = (i * 137.5) % 360  # Golden angle for good distribution
            saturation = 0.7 + (i % 3) * 0.1  # Vary saturation
            value = 0.8 + (i % 2) * 0.2  # Vary brightness
            rgb = colorsys.hsv_to_rgb(hue/360, saturation, value)
            rgb_int = [int(c * 255) for c in rgb]
            additional_colors.append(f'rgb({rgb_int[0]}, {rgb_int[1]}, {rgb_int[2]})')
        base_colors.extend(additional_colors)
    
    cluster_colors = base_colors[:n_clusters]
    
    # Ensure cluster column is treated as categorical
    df_plot['cluster'] = df_plot['cluster'].astype(str)
    
    # Create interactive plot with better color scheme
    fig = px.scatter(df_plot, 
                     x='x', y='y',
                     color='cluster',
                     hover_data=['feature_id', 'label'],
                     title=title,
                     labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'},
                     width=1000, height=700,
                     color_discrete_sequence=cluster_colors)
    
    # Update layout with better legend and styling
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
    
    # Update traces for better visibility with distinct colors
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
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{title}</h1>
            <div class="info">
                <strong>Interactive UMAP Visualization:</strong> Hover over points to see feature labels and cluster information. 
                Each cluster is assigned a distinct color for easy identification.
            </div>
            {html_content}
        </div>
    </body>
    </html>
    """
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(styled_html)
    
    return filename

def main():
    st.title("🧠 SAE Layer 16 - Interactive UMAP Visualization")
    st.markdown("**Hover over points to see feature labels and cluster information**")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📊 General Features", "💰 Financial Features"])
    
    st.sidebar.header("📊 Controls")
    
    # Number of clusters slider
    n_clusters = st.sidebar.slider(
        "Number of Clusters", 
        min_value=10, 
        max_value=100, 
        value=50, 
        step=5,
        help="Adjust the number of clusters for different granularity"
    )
    
    with tab1:
        st.subheader("📊 General Features")
        # Load data
        with st.spinner("Loading SAE vectors and labels..."):
            vectors, labels_df = load_data()
        
        # Create visualization
        fig, cluster_labels, feature_to_label = create_visualization(
            vectors, labels_df, n_clusters, "SAE Layer 16 - General Features Clustering"
        )
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Export button
        if st.button("📥 Export as HTML", key="export_general"):
            filename = save_html_export(fig, "SAE Layer 16 - General Features Clustering", "general_features_umap.html")
            st.success(f"✅ HTML file saved as: {filename}")
            st.info("You can now share this HTML file with others. It contains the full interactive visualization.")
        
        # Add color information
        st.info(f"🎨 **Color Coding**: Each cluster is assigned a distinct color from a high-contrast palette. The legend shows {len(set(cluster_labels))} different clusters with their unique colors for easy identification.")
        
        # Display cluster statistics
        st.subheader("📈 Cluster Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Total Features:** {len(vectors)}")
            st.write(f"**Total Clusters:** {len(set(cluster_labels))}")
            st.write(f"**Features with Labels:** {len(labels_df)}")
        
        with col2:
            # Cluster size distribution
            cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
            st.write("**Cluster Sizes:**")
            for cluster_id in sorted(set(cluster_labels)):
                size = cluster_sizes[cluster_id]
                st.write(f"- Cluster {cluster_id}: {size} features")
        
        # Display raw data
        if st.checkbox("Show Raw Data"):
            st.subheader("📋 Raw Data")
            # Create results for features that have labels
            results_data = []
            for i in range(len(vectors)):
                if i in feature_to_label:
                    results_data.append({
                        'feature': i,
                        'label': feature_to_label[i],
                        'cluster': cluster_labels[i]
                    })
            
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)
    
    with tab2:
        st.subheader("💰 Financial Features")
        # Load financial data
        with st.spinner("Loading SAE vectors and financial labels..."):
            vectors, labels_df = load_financial_data()
        
        # Create visualization
        fig, cluster_labels, feature_to_label = create_visualization(
            vectors, labels_df, n_clusters, "SAE Layer 16 - Financial Features Clustering"
        )
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Export button
        if st.button("📥 Export as HTML", key="export_financial"):
            filename = save_html_export(fig, "SAE Layer 16 - Financial Features Clustering", "financial_features_umap.html")
            st.success(f"✅ HTML file saved as: {filename}")
            st.info("You can now share this HTML file with others. It contains the full interactive visualization.")
        
        # Add color information
        st.info(f"🎨 **Color Coding**: Each cluster is assigned a distinct color from a high-contrast palette. The legend shows {len(set(cluster_labels))} different clusters with their unique colors for easy identification.")
        
        # Display cluster statistics
        st.subheader("📈 Cluster Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Total Features:** {len(vectors)}")
            st.write(f"**Total Clusters:** {len(set(cluster_labels))}")
            st.write(f"**Features with Labels:** {len(labels_df)}")
        
        with col2:
            # Cluster size distribution
            cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
            st.write("**Cluster Sizes:**")
            for cluster_id in sorted(set(cluster_labels)):
                size = cluster_sizes[cluster_id]
                st.write(f"- Cluster {cluster_id}: {size} features")
        
        # Display raw data
        if st.checkbox("Show Raw Data", key="financial"):
            st.subheader("📋 Raw Data")
            # Create results for features that have labels
            results_data = []
            for i in range(len(vectors)):
                if i in feature_to_label:
                    results_data.append({
                        'feature': i,
                        'label': feature_to_label[i],
                        'cluster': cluster_labels[i]
                    })
            
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)

if __name__ == "__main__":
    main()