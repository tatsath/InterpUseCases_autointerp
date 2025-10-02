"""
Create Clean Circuit Tracing Visualization
A cleaner version without emoji characters
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict

def create_clean_circuit_visualization():
    """Create a clean visualization of circuit tracing results."""
    
    # Set up the figure
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Circuit Tracing Analysis Results', fontsize=20, fontweight='bold')
    
    # Data from the circuit tracing results
    layers = [4, 10, 16, 22, 28]
    
    # Most active features per layer
    layer_features = {
        4: [41, 144, 39, 215, 139],
        10: [46, 192, 41, 144, 215],
        16: [48, 363, 215, 92, 39],
        22: [254, 37, 218, 213, 363],
        28: [93, 134, 219, 55, 218]
    }
    
    # Features that flow across multiple layers
    circuit_features = {
        56: [4, 10, 16, 22, 28],  # All 5 layers
        110: [4, 10, 16, 22],      # 4 layers
        139: [4, 10, 16, 22],      # 4 layers
        317: [4, 10, 22, 28],      # 4 layers
        15: [4, 10, 16, 22],       # 4 layers
        48: [4, 10, 16, 22],       # 4 layers
        39: [4, 16, 28],           # 3 layers
        215: [4, 10, 16],          # 3 layers
        363: [16, 22, 28],         # 3 layers
        92: [16, 22, 28]           # 3 layers
    }
    
    # 1. Circuit Flow Diagram
    ax1 = plt.subplot(2, 3, 1)
    create_circuit_flow_diagram(ax1, layers, circuit_features)
    
    # 2. Layer-specific Feature Heatmap
    ax2 = plt.subplot(2, 3, 2)
    create_layer_heatmap(ax2, layers, layer_features)
    
    # 3. Feature Activation Frequency
    ax3 = plt.subplot(2, 3, 3)
    create_activation_frequency_chart(ax3)
    
    # 4. Filtering Effectiveness
    ax4 = plt.subplot(2, 3, 4)
    create_filtering_chart(ax4)
    
    # 5. Feature Distribution Across Layers
    ax5 = plt.subplot(2, 3, 5)
    create_feature_distribution(ax5, circuit_features)
    
    # 6. Summary Statistics
    ax6 = plt.subplot(2, 3, 6)
    create_summary_stats(ax6)
    
    plt.tight_layout()
    plt.savefig('circuit_tracing_clean.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_circuit_flow_diagram(ax, layers, circuit_features):
    """Create a circuit flow diagram showing feature connections across layers."""
    ax.set_title('Circuit Flow Across Layers', fontsize=14, fontweight='bold')
    
    # Draw layers as vertical lines
    layer_positions = {layer: i for i, layer in enumerate(layers)}
    
    for i, layer in enumerate(layers):
        ax.axvline(x=i, ymin=0, ymax=1, color='black', linewidth=2, alpha=0.7)
        ax.text(i, -0.05, f'Layer {layer}', ha='center', va='top', fontweight='bold')
    
    # Draw feature flows
    colors = plt.cm.Set3(np.linspace(0, 1, len(circuit_features)))
    
    for i, (feature, layer_list) in enumerate(circuit_features.items()):
        color = colors[i]
        y_pos = 0.9 - (i * 0.08)  # Spread features vertically
        
        # Draw horizontal lines for features
        for j in range(len(layer_list) - 1):
            start_layer = layer_list[j]
            end_layer = layer_list[j + 1]
            start_x = layer_positions[start_layer]
            end_x = layer_positions[end_layer]
            
            ax.plot([start_x, end_x], [y_pos, y_pos], 
                   color=color, linewidth=3, alpha=0.8)
            
            # Add arrow
            ax.annotate('', xy=(end_x, y_pos), xytext=(start_x, y_pos),
                       arrowprops=dict(arrowstyle='->', color=color, lw=2))
        
        # Add feature label
        ax.text(-0.1, y_pos, f'F{feature}', ha='right', va='center', 
               fontweight='bold', color=color)
    
    ax.set_xlim(-0.3, len(layers) - 0.7)
    ax.set_ylim(-0.1, 1.0)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f'L{layer}' for layer in layers])
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)

def create_layer_heatmap(ax, layers, layer_features):
    """Create a heatmap showing feature activity across layers."""
    ax.set_title('Feature Activity Heatmap', fontsize=14, fontweight='bold')
    
    # Create matrix for heatmap
    all_features = set()
    for features in layer_features.values():
        all_features.update(features)
    all_features = sorted(list(all_features))
    
    matrix = np.zeros((len(all_features), len(layers)))
    
    for i, layer in enumerate(layers):
        for j, feature in enumerate(all_features):
            if feature in layer_features[layer]:
                # Higher values for features that appear earlier in the list (more active)
                rank = layer_features[layer].index(feature)
                matrix[j, i] = len(layer_features[layer]) - rank
    
    # Create heatmap
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    
    # Set labels
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f'L{layer}' for layer in layers])
    ax.set_yticks(range(len(all_features)))
    ax.set_yticklabels([f'F{feature}' for feature in all_features])
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Activity Level')
    
    # Rotate y-axis labels
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

def create_activation_frequency_chart(ax):
    """Create a bar chart showing feature activation frequencies."""
    ax.set_title('Top Features by Activation Frequency', fontsize=14, fontweight='bold')
    
    # Data from results
    features = [317, 56, 39, 215, 213, 264, 41, 110, 139, 144, 15, 218, 55, 46, 192]
    frequencies = [20.0, 20.0, 18.0, 18.0, 18.0, 18.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 14.0, 14.0, 14.0]
    
    bars = ax.bar(range(len(features)), frequencies, color='skyblue', alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, freq) in enumerate(zip(bars, frequencies)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{freq:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Features')
    ax.set_ylabel('Activation Frequency (%)')
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels([f'F{feature}' for feature in features], rotation=45)
    ax.grid(True, alpha=0.3)

def create_filtering_chart(ax):
    """Create a chart showing filtering effectiveness."""
    ax.set_title('Filtering Effectiveness', fontsize=14, fontweight='bold')
    
    categories = ['Always-on\nFeatures', 'Selective\nFeatures']
    counts = [50, 27]
    colors = ['lightcoral', 'lightgreen']
    
    bars = ax.bar(categories, counts, color=colors, alpha=0.7)
    
    # Add percentage labels
    total = sum(counts)
    for bar, count in zip(bars, counts):
        percentage = (count / total) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', 
               fontweight='bold', fontsize=12)
    
    ax.set_ylabel('Number of Features')
    ax.set_ylim(0, max(counts) * 1.2)
    ax.grid(True, alpha=0.3)

def create_feature_distribution(ax, circuit_features):
    """Create a chart showing feature distribution across layers."""
    ax.set_title('Features by Layer Count', fontsize=14, fontweight='bold')
    
    # Count features by number of layers they appear in
    layer_counts = defaultdict(int)
    for feature, layers in circuit_features.items():
        layer_counts[len(layers)] += 1
    
    counts = list(layer_counts.keys())
    frequencies = list(layer_counts.values())
    
    bars = ax.bar([f'{count} layers' for count in counts], frequencies, 
                 color='lightblue', alpha=0.7)
    
    # Add value labels
    for bar, freq in zip(bars, frequencies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               str(freq), ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Number of Layers')
    ax.set_ylabel('Number of Features')
    ax.grid(True, alpha=0.3)

def create_summary_stats(ax):
    """Create a summary statistics panel."""
    ax.set_title('Summary Statistics', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    stats_text = """FILTERING RESULTS:
• Always-on features filtered: 50
• Selective features analyzed: 27
• Filtering effectiveness: 64.9%

CIRCUIT ANALYSIS:
• Features with circuit flow: 18
• Most active feature: 317 (20%)
• Features across all layers: 1

LAYER COVERAGE:
• Layer 4: 5 top features
• Layer 10: 5 top features  
• Layer 16: 5 top features
• Layer 22: 5 top features
• Layer 28: 5 top features

KEY INSIGHTS:
• Successfully filtered always-on features
• Identified meaningful circuit flows
• Layer-specific feature patterns detected
• Focus on selective, prompt-varying features"""
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

def main():
    """Main function to create the visualization."""
    print("Creating clean circuit tracing visualization...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    create_clean_circuit_visualization()
    
    print("Visualization saved as 'circuit_tracing_clean.png'")

if __name__ == "__main__":
    main()
