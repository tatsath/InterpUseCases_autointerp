"""
Circuit Visualization and Analysis Utilities

This module provides comprehensive visualization and analysis tools for
financial circuit tracing results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Tuple, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

class CircuitVisualizer:
    """Visualization utilities for circuit tracing results."""
    
    def __init__(self, results_dir: str = "circuit_tracing_results"):
        self.results_dir = results_dir
        self.setup_plotting()
    
    def setup_plotting(self):
        """Setup matplotlib and seaborn plotting parameters."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def plot_circuit_paths(self, circuit_paths: List[Dict], title: str = "Circuit Paths", save_path: Optional[str] = None):
        """Plot circuit paths as a network diagram."""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create a directed graph for visualization
        G = nx.DiGraph()
        
        # Add nodes and edges from circuit paths
        for i, path_data in enumerate(circuit_paths):
            path = path_data["path"]
            weight = path_data["weight_product"]
            
            # Add nodes
            for layer, feature in path:
                G.add_node((layer, feature), layer=layer, feature=feature)
            
            # Add edges
            for j in range(len(path) - 1):
                u, v = path[j], path[j + 1]
                G.add_edge(u, v, weight=weight, path_id=i)
        
        # Position nodes
        pos = {}
        layers = sorted(set(node[0] for node in G.nodes()))
        for layer in layers:
            layer_nodes = [node for node in G.nodes() if node[0] == layer]
            for i, node in enumerate(layer_nodes):
                pos[node] = (layer, i)
        
        # Draw nodes
        node_colors = [G.nodes[node].get('layer', 0) for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              cmap=plt.cm.viridis, node_size=300, alpha=0.8)
        
        # Draw edges with different colors for different paths
        edge_colors = [G[u][v].get('path_id', 0) for u, v in G.edges()]
        # Convert path IDs to colors
        unique_paths = list(set(edge_colors))
        color_map = plt.cm.Set3(np.linspace(0, 1, len(unique_paths)))
        edge_color_list = [color_map[unique_paths.index(path_id)] for path_id in edge_colors]
        nx.draw_networkx_edges(G, pos, edge_color=edge_color_list, 
                              alpha=0.6, arrows=True, arrowsize=20)
        
        # Add labels
        labels = {node: f"L{node[0]}:F{node[1]}" for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Feature Index", fontsize=12)
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=plt.cm.viridis(i/len(layers)), 
                                     markersize=10, label=f'Layer {layer}') 
                          for i, layer in enumerate(layers)]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_activation_heatmap(self, activations: Dict, title: str = "Feature Activation Heatmap", save_path: Optional[str] = None):
        """Plot feature activation heatmap across layers."""
        # Prepare data for heatmap
        layers = sorted(activations.keys())
        all_features = set()
        
        for layer_activations in activations.values():
            if isinstance(layer_activations, np.ndarray):
                all_features.update(range(layer_activations.shape[-1]))
        
        all_features = sorted(list(all_features))
        
        # Create activation matrix
        activation_matrix = np.zeros((len(layers), len(all_features)))
        
        for i, layer in enumerate(layers):
            layer_acts = activations[layer]
            if isinstance(layer_acts, np.ndarray):
                # Take max activation across sequence length
                max_acts = layer_acts.max(axis=1).squeeze(0)  # [F]
                for j, feat in enumerate(all_features):
                    if feat < len(max_acts):
                        activation_matrix[i, j] = max_acts[feat]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(16, 8))
        im = ax.imshow(activation_matrix, cmap='viridis', aspect='auto')
        
        # Set labels
        ax.set_xticks(range(len(all_features)))
        ax.set_xticklabels([f"F{f}" for f in all_features], rotation=45)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f"Layer {l}" for l in layers])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Activation Strength', rotation=270, labelpad=20)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Feature Index', fontsize=12)
        ax.set_ylabel('Layer', fontsize=12)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_circuit_strength_comparison(self, results: Dict[str, Any], save_path: Optional[str] = None):
        """Plot circuit strength comparison across topics."""
        topic_data = []
        
        for topic, topic_results in results.items():
            if isinstance(topic_results, dict):
                for prompt_key, prompt_data in topic_results.items():
                    if isinstance(prompt_data, dict) and "circuit_paths" in prompt_data:
                        for i, path in enumerate(prompt_data["circuit_paths"]):
                            topic_data.append({
                                "Topic": topic,
                                "Prompt": prompt_key,
                                "Path": i + 1,
                                "Strength": path["weight_product"],
                                "Length": len(path["path"])
                            })
        
        if not topic_data:
            print("No circuit data found for visualization")
            return
        
        df = pd.DataFrame(topic_data)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Circuit strength by topic
        sns.boxplot(data=df, x="Topic", y="Strength", ax=ax1)
        ax1.set_title("Circuit Strength by Topic", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Topic", fontsize=12)
        ax1.set_ylabel("Circuit Strength", fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # Circuit length by topic
        sns.boxplot(data=df, x="Topic", y="Length", ax=ax2)
        ax2.set_title("Circuit Length by Topic", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Topic", fontsize=12)
        ax2.set_ylabel("Circuit Length", fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_circuit_plot(self, circuit_paths: List[Dict], title: str = "Interactive Circuit Paths"):
        """Create an interactive Plotly visualization of circuit paths."""
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        for i, path_data in enumerate(circuit_paths):
            path = path_data["path"]
            weight = path_data["weight_product"]
            
            for layer, feature in path:
                G.add_node((layer, feature), layer=layer, feature=feature)
            
            for j in range(len(path) - 1):
                u, v = path[j], path[j + 1]
                G.add_edge(u, v, weight=weight, path_id=i)
        
        # Position nodes
        pos = {}
        layers = sorted(set(node[0] for node in G.nodes()))
        for layer in layers:
            layer_nodes = [node for node in G.nodes() if node[0] == layer]
            for i, node in enumerate(layer_nodes):
                pos[node] = (layer, i)
        
        # Extract edge information
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_info.append(f"L{edge[0][0]}:F{edge[0][1]} → L{edge[1][0]}:F{edge[1][1]}")
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='rgba(125,125,125,0.5)'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Extract node information
        node_x = []
        node_y = []
        node_text = []
        node_hover = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"L{node[0]}:F{node[1]}")
            node_hover.append(f"Layer: {node[0]}<br>Feature: {node[1]}")
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            hovertext=node_hover,
            marker=dict(
                size=20,
                color=[node[0] for node in G.nodes()],
                colorscale='viridis',
                line=dict(width=2, color='black')
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=dict(text=title, font=dict(size=16)),
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Interactive Circuit Path Visualization",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color="black", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=True, title="Layer"),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="Feature Index"),
                           plot_bgcolor='white'
                       ))
        
        return fig
    
    def generate_circuit_report(self, results: Dict[str, Any], output_path: str = "circuit_analysis_report.html"):
        """Generate a comprehensive HTML report of circuit analysis."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Financial Circuit Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .circuit-path { background-color: #f9f9f9; padding: 10px; margin: 5px 0; border-left: 4px solid #007acc; }
                .stats { display: flex; justify-content: space-around; }
                .stat-box { text-align: center; padding: 10px; background-color: #e8f4f8; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Financial Circuit Analysis Report</h1>
                <p>Generated on: {timestamp}</p>
            </div>
        """
        
        # Add analysis for each topic
        for topic, topic_results in results.items():
            if isinstance(topic_results, dict):
                html_content += f"""
                <div class="section">
                    <h2>Topic: {topic.replace('_', ' ').title()}</h2>
                """
                
                for prompt_key, prompt_data in topic_results.items():
                    if isinstance(prompt_data, dict) and "circuit_paths" in prompt_data:
                        html_content += f"""
                        <h3>Prompt: {prompt_key}</h3>
                        <p><strong>Text:</strong> {prompt_data.get('prompt', 'N/A')[:200]}...</p>
                        """
                        
                        if "graph_stats" in prompt_data:
                            stats = prompt_data["graph_stats"]
                            html_content += f"""
                            <div class="stats">
                                <div class="stat-box">
                                    <h4>Graph Nodes</h4>
                                    <p>{stats.get('nodes', 0)}</p>
                                </div>
                                <div class="stat-box">
                                    <h4>Graph Edges</h4>
                                    <p>{stats.get('edges', 0)}</p>
                                </div>
                            </div>
                            """
                        
                        html_content += "<h4>Circuit Paths:</h4>"
                        for i, path in enumerate(prompt_data["circuit_paths"]):
                            path_str = " → ".join([f"L{layer}:F{feature}" for layer, feature in path["path"]])
                            html_content += f"""
                            <div class="circuit-path">
                                <strong>Path {i+1}:</strong> {path_str}<br>
                                <strong>Strength:</strong> {path['weight_product']:.4f}<br>
                                <strong>Edge Types:</strong> {', '.join(path['edge_kinds'])}
                            </div>
                            """
                
                html_content += "</div>"
        
        html_content += """
        </body>
        </html>
        """
        
        # Replace timestamp
        from datetime import datetime
        html_content = html_content.replace("{timestamp}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Circuit analysis report saved to {output_path}")

class CircuitAnalyzer:
    """Analysis utilities for circuit tracing results."""
    
    def __init__(self, results_dir: str = "circuit_tracing_results"):
        self.results_dir = results_dir
    
    def load_results(self) -> Dict[str, Any]:
        """Load circuit tracing results from JSON files."""
        results = {}
        
        # Load main results file
        main_file = os.path.join(self.results_dir, "circuit_paths.json")
        if os.path.exists(main_file):
            with open(main_file, 'r') as f:
                results = json.load(f)
        
        return results
    
    def analyze_circuit_patterns(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns in circuit tracing results."""
        analysis = {
            "topic_stats": {},
            "feature_frequency": {},
            "layer_transitions": {},
            "circuit_strengths": {}
        }
        
        for topic, topic_results in results.items():
            if not isinstance(topic_results, dict):
                continue
            
            topic_analysis = {
                "total_prompts": 0,
                "total_circuits": 0,
                "avg_circuit_strength": 0,
                "avg_circuit_length": 0,
                "unique_features": set(),
                "layer_coverage": set()
            }
            
            circuit_strengths = []
            circuit_lengths = []
            
            for prompt_key, prompt_data in topic_results.items():
                if isinstance(prompt_data, dict) and "circuit_paths" in prompt_data:
                    topic_analysis["total_prompts"] += 1
                    
                    for path in prompt_data["circuit_paths"]:
                        topic_analysis["total_circuits"] += 1
                        circuit_strengths.append(path["weight_product"])
                        circuit_lengths.append(len(path["path"]))
                        
                        # Track features and layers
                        for layer, feature in path["path"]:
                            topic_analysis["unique_features"].add((layer, feature))
                            topic_analysis["layer_coverage"].add(layer)
            
            if circuit_strengths:
                topic_analysis["avg_circuit_strength"] = np.mean(circuit_strengths)
                topic_analysis["avg_circuit_length"] = np.mean(circuit_lengths)
            
            analysis["topic_stats"][topic] = topic_analysis
        
        return analysis
    
    def find_common_features(self, results: Dict[str, Any]) -> Dict[str, List[Tuple[int, int]]]:
        """Find features that appear frequently across different topics."""
        feature_counts = {}
        
        for topic, topic_results in results.items():
            if not isinstance(topic_results, dict):
                continue
            
            for prompt_key, prompt_data in topic_results.items():
                if isinstance(prompt_data, dict) and "circuit_paths" in prompt_data:
                    for path in prompt_data["circuit_paths"]:
                        for layer, feature in path["path"]:
                            key = (layer, feature)
                            feature_counts[key] = feature_counts.get(key, 0) + 1
        
        # Sort by frequency
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "most_common": sorted_features[:20],
            "feature_frequency": dict(sorted_features)
        }
    
    def export_analysis_summary(self, analysis: Dict[str, Any], output_path: str = "circuit_analysis_summary.json"):
        """Export analysis summary to JSON file."""
        # Convert sets to lists for JSON serialization
        def convert_sets(obj):
            if isinstance(obj, dict):
                return {k: convert_sets(v) for k, v in obj.items()}
            elif isinstance(obj, set):
                return list(obj)
            else:
                return obj
        
        analysis_serializable = convert_sets(analysis)
        
        with open(output_path, 'w') as f:
            json.dump(analysis_serializable, f, indent=2)
        
        print(f"Analysis summary exported to {output_path}")

def main():
    """Main function for visualization and analysis."""
    # Initialize visualizer and analyzer
    visualizer = CircuitVisualizer()
    analyzer = CircuitAnalyzer()
    
    # Load results
    results = analyzer.load_results()
    if not results:
        print("No results found. Please run the circuit tracer first.")
        return
    
    # Analyze patterns
    analysis = analyzer.analyze_circuit_patterns(results)
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # Plot circuit strength comparison
    visualizer.plot_circuit_strength_comparison(results, "circuit_strength_comparison.png")
    
    # Generate HTML report
    visualizer.generate_circuit_report(results, "circuit_analysis_report.html")
    
    # Export analysis summary
    analyzer.export_analysis_summary(analysis, "circuit_analysis_summary.json")
    
    print("Visualization and analysis complete!")

if __name__ == "__main__":
    main()
