"""
Create a visual file structure diagram for the Circuit Tracer system
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_file_structure_diagram():
    """Create a visual file structure diagram."""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'Circuit Tracer System - File Structure', 
            ha='center', va='center', fontsize=20, fontweight='bold')
    
    # Define file categories with colors
    categories = {
        'CORE IMPLEMENTATION': {
            'color': '#FF6B6B',
            'files': [
                'comprehensive_labeled_tracer.py ⭐',
                'financial_circuit_tracer.py',
                'simple_fast_tracer.py',
                'simple_tracer.py'
            ]
        },
        'VISUALIZATION': {
            'color': '#4ECDC4',
            'files': [
                'visualize_circuit_results.py ⭐',
                'create_clean_visualization.py',
                'circuit_visualization.py',
                'circuit_tracing_clean.png ⭐'
            ]
        },
        'WEB APPLICATIONS': {
            'color': '#45B7D1',
            'files': [
                'Text_Tracing_app.py ⭐',
                'Reply_Tracing_app.py ⭐',
                'test_interactive_circuit.html'
            ]
        },
        'TESTING & DEBUG': {
            'color': '#96CEB4',
            'files': [
                'test_circuit_tracer.py ⭐',
                'debug_raw_activations.py',
                'complete_analysis_tracer.py',
                'simple_circuit_test.py'
            ]
        },
        'DOCUMENTATION': {
            'color': '#FFEAA7',
            'files': [
                'CIRCUIT_TRACER_README.md ⭐',
                'README.md',
                'CIRCUIT_TRACER_FILE_OVERVIEW.md ⭐',
                'SAE_Training_Metrics_Guide.md'
            ]
        },
        'CONFIGURATION': {
            'color': '#DDA0DD',
            'files': [
                'circuit_tracer_requirements.txt ⭐',
                'circuit_tracing_results.txt',
                'requirements.txt'
            ]
        }
    }
    
    # Draw categories
    y_pos = 10.5
    x_pos = 0.5
    
    for i, (category, data) in enumerate(categories.items()):
        # Category box
        box = FancyBboxPatch((x_pos, y_pos-0.8), 4, 0.6, 
                            boxstyle="round,pad=0.02", 
                            facecolor=data['color'], 
                            edgecolor='black', 
                            linewidth=2)
        ax.add_patch(box)
        
        # Category title
        ax.text(x_pos + 2, y_pos-0.5, category, 
                ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Files in category
        for j, file in enumerate(data['files']):
            file_y = y_pos - 1.2 - (j * 0.25)
            
            # File box
            file_box = FancyBboxPatch((x_pos + 0.1, file_y-0.1), 3.8, 0.2, 
                                     boxstyle="round,pad=0.01", 
                                     facecolor='white', 
                                     edgecolor=data['color'], 
                                     linewidth=1)
            ax.add_patch(file_box)
            
            # File name
            ax.text(x_pos + 0.2, file_y, file, 
                    ha='left', va='center', fontsize=9)
        
        # Move to next column
        x_pos += 4.5
        if x_pos > 8:
            x_pos = 0.5
            y_pos -= 4
    
    # Add legend
    legend_y = 2
    ax.text(0.5, legend_y, 'LEGEND:', fontsize=14, fontweight='bold')
    
    legend_items = [
        ('⭐', 'Primary/Recommended Files'),
        ('✅', 'Production Ready'),
        ('⚠️', 'Legacy/Experimental'),
        ('🧪', 'Testing/Debug Tools')
    ]
    
    for i, (symbol, description) in enumerate(legend_items):
        ax.text(0.5, legend_y - 0.3 - (i * 0.2), f'{symbol} {description}', 
                fontsize=10, ha='left')
    
    # Add usage instructions
    instructions = """
    QUICK START:
    1. pip install -r circuit_tracer_requirements.txt
    2. python comprehensive_labeled_tracer.py
    3. streamlit run Text_Tracing_app.py
    4. View circuit_tracing_clean.png
    """
    
    ax.text(5.5, 2, instructions, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('circuit_tracer_file_structure.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to create the file structure diagram."""
    print("Creating file structure diagram...")
    create_file_structure_diagram()
    print("File structure diagram saved as 'circuit_tracer_file_structure.png'")

if __name__ == "__main__":
    main()
