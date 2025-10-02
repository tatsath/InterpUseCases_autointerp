# Financial Circuit Tracer

A comprehensive system for analyzing feature relationships across multiple Sparse Autoencoder (SAE) layers in financial contexts using circuit tracing techniques.

## 🎯 Overview

The Financial Circuit Tracer analyzes how different SAE features interact across transformer layers when processing financial text. It builds feature graphs, traces circuit paths, and provides detailed visualizations to understand the internal reasoning patterns of language models on financial topics.

## 🚀 Key Features

- **Multi-Layer SAE Analysis**: Analyze feature activations across multiple SAE layers (4, 10, 16, 22, 28)
- **Circuit Tracing**: Find the most important feature paths from early to late layers
- **Financial Topic Focus**: Pre-built datasets for 5 key financial topics
- **Attention-Mediated Connections**: Use attention weights to understand feature relationships
- **Comprehensive Visualizations**: Interactive plots, heatmaps, and network diagrams
- **Export Capabilities**: JSON exports for further analysis

## 📁 Files

- **`financial_circuit_tracer.py`** - Main circuit tracing implementation
- **`circuit_visualization.py`** - Visualization and analysis utilities
- **`test_circuit_tracer.py`** - Test script for functionality verification
- **`circuit_tracer_requirements.txt`** - Python dependencies

## 🎯 Financial Topics Covered

1. **Mergers & Acquisitions**: M&A deals, corporate consolidation, regulatory approval
2. **Financial Leaders**: CEO statements, executive leadership, corporate governance
3. **Financial Entities**: Banks, hedge funds, investment firms, regulatory bodies
4. **Financial Regulations**: SEC rules, Basel III, compliance, policy changes
5. **Market Events & Sentiments**: Market crashes, volatility, investor sentiment, economic cycles

## 🔧 Installation

```bash
# Install dependencies
pip install -r circuit_tracer_requirements.txt

# Ensure you have the SAE models in the correct path
# Default path: /home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun
```

## 🚀 Quick Start

### Basic Usage

```python
from financial_circuit_tracer import FinancialCircuitTracer

# Initialize tracer
tracer = FinancialCircuitTracer(
    model_path="meta-llama/Llama-2-7b-hf",
    sae_path="/path/to/sae/models",
    device="cuda"
)

# Analyze a single prompt
circuit_paths, graph, activations = tracer.trace_circuits_for_prompt(
    prompt="Jamie Dimon's leadership at JPMorgan Chase...",
    topic="financial_leaders",
    start_layer=4,
    end_layer=28,
    k_paths=5
)

# Print results
for i, path in enumerate(circuit_paths):
    path_str = " -> ".join([f"L{layer}:F{feature}" for layer, feature in path["path"]])
    print(f"Path {i+1}: {path_str} (strength: {path['weight_product']:.4f})")
```

### Topic Analysis

```python
# Analyze all prompts for a specific topic
results = tracer.analyze_topic("mergers_acquisitions")

# Export results
tracer.export_results(results, "ma_analysis_results")
```

### Visualization

```python
from circuit_visualization import CircuitVisualizer

# Create visualizations
visualizer = CircuitVisualizer()
visualizer.plot_circuit_paths(circuit_paths, "M&A Circuit Analysis")
visualizer.plot_circuit_strength_comparison(results)
```

## 🧪 Testing

Run the test suite to verify functionality:

```bash
python test_circuit_tracer.py
```

This will test:
- Single prompt analysis
- Topic-based analysis
- Visualization capabilities

## 📊 Understanding the Output

### Circuit Paths
- **Format**: `L{layer}:F{feature}` (e.g., `L4:F25 -> L10:F83 -> L16:F214`)
- **Strength**: Product of edge weights along the path (higher = more important)
- **Edge Types**: `attn_mediated`, `lagged`, `same_layer_corr`

### Feature Graph
- **Nodes**: `(layer, feature_id)` pairs
- **Edges**: Weighted connections between features
- **Weights**: Normalized to [0,1] range

### Analysis Results
- **Circuit Paths**: Top-k most important feature sequences
- **Graph Statistics**: Node/edge counts, connectivity metrics
- **Activation Data**: Raw feature activations per layer

## 🎯 Sample Prompts

### Mergers & Acquisitions
```
"Microsoft's acquisition of Activision Blizzard for $68.7 billion represents the largest gaming industry merger in history, reshaping the competitive landscape and regulatory environment."
```

### Financial Leaders
```
"Jamie Dimon's leadership at JPMorgan Chase has been marked by strategic acquisitions, digital transformation initiatives, and navigating multiple financial crises while maintaining strong capital ratios."
```

### Financial Regulations
```
"The SEC's new climate disclosure rules require public companies to report greenhouse gas emissions and climate-related risks, significantly impacting corporate reporting standards."
```

## 🔧 Configuration

### Model Configuration
- **Model**: `meta-llama/Llama-2-7b-hf`
- **SAE Layers**: 4, 10, 16, 22, 28
- **Features per Layer**: 400
- **Device**: Auto-detects CUDA availability

### Circuit Tracing Parameters
- **Attention Weight**: 0.6 (weight for attention-mediated connections)
- **Lag Weight**: 0.3 (weight for same-token lagged connections)
- **Same Layer Weight**: 0.1 (weight for same-layer correlations)
- **Top K Paths**: 5 (number of best paths to return)

## 📈 Advanced Usage

### Custom Financial Topics

```python
# Add custom topic to FinancialTopicData
tracer.topic_data.topics["custom_topic"] = {
    "keywords": ["keyword1", "keyword2"],
    "prompts": ["Custom prompt 1", "Custom prompt 2"]
}
```

### Custom SAE Layers

```python
# Modify the layers list in FinancialCircuitTracer.__init__
tracer.layers = [4, 8, 16, 24, 32]  # Custom layer selection
```

### Export Options

```python
# Export specific analysis
tracer.export_results(results, "custom_output_dir")

# Generate HTML report
from circuit_visualization import CircuitVisualizer
visualizer = CircuitVisualizer()
visualizer.generate_circuit_report(results, "custom_report.html")
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size or use CPU
   - Process prompts individually

2. **SAE Model Not Found**
   - Check SAE path configuration
   - Ensure safetensors files exist

3. **No Circuit Paths Found**
   - Check if start/end layers have valid features
   - Verify prompt contains relevant keywords

### Debug Mode

```python
# Enable debug output
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
tracer.trace_circuits_for_prompt(prompt, verbose=True)
```

## 📚 Technical Details

### Feature Graph Construction
1. **Nodes**: All SAE features across all layers
2. **Attention-Mediated Edges**: Features connected via attention weights
3. **Lagged Edges**: Same-token features across consecutive layers
4. **Same-Layer Edges**: Correlated features within the same layer

### Circuit Path Finding
1. **Start Features**: Top-k activated features in early layers on keyword tokens
2. **End Features**: Top-k activated features in late layers on final tokens
3. **Path Search**: Dijkstra's algorithm with log-transformed weights
4. **Ranking**: Paths ranked by product of edge weights

### Performance Optimization
- **Subsampling**: Random sampling for large feature spaces
- **Caching**: SAE weights cached in memory
- **Batch Processing**: Efficient tensor operations
- **Memory Management**: CPU offloading for large graphs

## 🤝 Contributing

To add new features or improve the circuit tracer:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is part of the InterpUseCases_autointerp repository.

## 🔗 Related Work

- **SAE Training**: Based on the SAE models in the parent directory
- **Feature Analysis**: Extends the Text_Tracing_app.py and Reply_Tracing_app.py
- **Circuit Tracing**: Inspired by mechanistic interpretability research

---

For questions or issues, please refer to the main repository documentation or create an issue.
