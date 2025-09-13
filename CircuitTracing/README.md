# Feature Activation Tracker

## 🔍 Overview
Interactive feature activation tracking and cross-layer correlation analysis for the finetuned Llama2-7b-Finance model using SAE (Sparse Autoencoder) features. This app tracks feature activations across all layers and analyzes correlations between features to understand circuit patterns and feature relationships.

## 🚀 Quick Start

### 1. Run the Feature Activation Tracker
```bash
conda activate sae
streamlit run feature_activation_tracker.py --server.port 8502
```

### 2. Access the App
Open your browser and go to `http://localhost:8502`

## 🎯 Features

### Core Functionality
- **Top Feature Activation Tracking**: Shows the most activated features for each prompt across all SAE layers (4, 10, 16, 22, 28)
- **Interactive Bar Charts**: Beautiful horizontal bar charts showing activation levels
- **Layer Comparison**: Side-by-side comparison of top features across layers
- **Activation Heatmap**: Visual heatmap showing activation patterns
- **Finetuned Model Labels**: Uses actual feature labels from finetuned model analysis
- **F1 Score Integration**: Shows feature performance scores from model analysis
- **Specialization Metrics**: Displays feature specialization levels

### Visualizations
- **Horizontal Bar Charts**: Beautiful bar charts showing top activated features for each layer
- **Layer Comparison Chart**: Grouped bar chart comparing top features across layers
- **Activation Heatmap**: Visual heatmap showing activation patterns across layers and features
- **Top Features Display**: Shows top activated features for each layer

## 🔧 How It Works

### 1. Feature Activation Tracking
For each input text, the app:
1. Processes the text through the model
2. Extracts hidden states from each SAE layer
3. Computes SAE activations for all features in each layer
4. Tracks activation patterns across layers

### 2. Correlation Analysis
The app analyzes:
- **Cross-layer correlations**: How features in different layers correlate
- **Feature relationships**: Which features work together across layers
- **Circuit patterns**: Understanding of neural pathways

### 3. Visualization
- **Heatmaps**: Show activation patterns across layers and features
- **Network graphs**: Display feature relationships and correlations
- **Evolution plots**: Track how features change across layers

## 📊 Analysis Types

### 1. Activation Heatmap
- Shows feature activation strength across all layers
- Helps identify which features are most active
- Reveals layer-specific activation patterns

### 2. Feature Evolution
- Tracks how specific features change across layers
- Shows feature development and specialization
- Identifies feature hierarchies

### 3. Correlation Network
- Network visualization of feature relationships
- Shows which features correlate across layers
- Helps understand circuit patterns

### 4. Top Features Analysis
- **Horizontal Bar Charts**: Beautiful bar charts showing top activated features for each layer
- **Layer Comparison**: Side-by-side comparison of top features across layers
- **Feature Details**: Shows feature labels, F1 scores, and specialization metrics
- **Interactive Tables**: Detailed feature information with hover tooltips

## 🎛️ Controls

### Input
- **Text Input**: Enter financial text to analyze
- **Analysis Options**: Configure correlation threshold and number of top features

### Parameters
- **Correlation Threshold**: Minimum correlation for network connections (0.1-0.9)
- **Top Features**: Number of top features to display (5-20)

## 📈 Use Cases

### 1. Circuit Analysis
- Understand how features work together across layers
- Identify key feature pathways in the model
- Analyze feature specialization and development

### 2. Model Understanding
- Gain insights into model architecture
- Understand feature hierarchies
- Analyze model behavior on specific inputs

### 3. Feature Engineering
- Identify important features for specific tasks
- Understand feature relationships
- Guide feature selection for steering

## 🔧 Technical Details

### Model and SAE
- **Model**: `cxllin/Llama2-7b-Finance`
- **SAE**: `llama2_7b_finance_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun`
- **Layers**: 4, 10, 16, 22, 28
- **Features per Layer**: 400

### Analysis Methods
- **Activation Tracking**: Direct SAE activation computation
- **Correlation Analysis**: Pearson correlation coefficients
- **Network Analysis**: Graph-based feature relationship analysis
- **Visualization**: Interactive plots using Plotly

## 🎯 Usage Tips

1. **Start with financial text** for best results
2. **Adjust correlation threshold** to see more/fewer connections
3. **Use different text inputs** to see how features respond
4. **Compare results** across different inputs
5. **Focus on high-activation features** for circuit analysis

## 📁 Files

- **`feature_activation_tracker.py`** - Main Streamlit application
- **`circuit_tracer.py`** - Original circuit tracing module
- **`requirements.txt`** - Python dependencies
- **`README.md`** - This documentation

## 🔗 Integration

This app integrates with the existing Steering app:
- Uses the same model and SAE weights
- Shares feature data and labels
- Complements steering analysis with activation tracking
- Provides deeper insights into feature behavior

## 🎯 Next Steps

1. **Run the app** and analyze your financial text
2. **Explore correlations** between features across layers
3. **Identify key features** for your specific use case
4. **Use insights** to improve feature steering
5. **Analyze different texts** to understand feature patterns
