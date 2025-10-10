# Llama-2-7b-hf SAE Trading Analysis

This directory contains Python scripts that integrate Llama-2-7b-hf SAE features with trading analysis, based on the `ActivationDetectTwoPass.ipynb` notebook example.

## Files

### 1. `llama2_sae_trading_analysis.py`
**Comprehensive trading analysis script**
- Full-featured analyzer with SAE model loading
- Individual text analysis
- Trading dataset creation
- Model training and evaluation
- Market sentiment analysis

### 2. `llama2_sae_simple_trading.py`
**Simplified version for basic analysis**
- Lightweight analyzer
- Focus on core SAE functionality
- Basic text analysis and feature extraction
- Market sentiment analysis

### 3. `llama2_sae_integrated_trading.py`
**Integrated version with existing trading infrastructure**
- Follows the notebook's approach closely
- Headline analysis and feature aggregation
- Feature matrix building
- Trading model training
- Comprehensive sentiment analysis

## Requirements

```bash
pip install torch transformers safetensors pandas numpy scikit-learn
```

## Usage

### Basic Usage
```python
from llama2_sae_simple_trading import LlamaSAESimpleTrading

# Initialize analyzer
analyzer = LlamaSAESimpleTrading()

# Analyze financial text
analysis = analyzer.analyze_financial_text("Apple reports record quarterly earnings")
print(analysis)
```

### Advanced Usage
```python
from llama2_sae_integrated_trading import LlamaSAEIntegratedTrading

# Initialize analyzer
analyzer = LlamaSAEIntegratedTrading()

# Analyze headlines for features
headlines = ["Apple reports earnings", "Tesla stock surges"]
top_features = analyzer.analyze_headlines_for_features(headlines)

# Build feature matrix
returns = [0.05, 0.15]
selected_features = top_features.head(10)['feature_id'].tolist()
feature_df = analyzer.build_feature_matrix(headlines, returns, selected_features)

# Train trading model
model_results = analyzer.train_trading_model(feature_df)
print(f"Model accuracy: {model_results['accuracy']:.4f}")
```

## Key Features

### SAE Model Integration
- Loads pre-trained Llama-2-7b-hf SAE model from layer 16
- Extracts activations for financial text analysis
- Maps features to financial concepts (earnings, revenue, trading, etc.)

### Financial Concept Mapping
The analyzer maps SAE features to financial concepts:
- **Earnings Reports**: Features 332, 105, 214
- **Revenue Metrics**: Features 214, 66, 181
- **Stock Performance**: Features 66, 133, 267
- **Trading Strategies**: Features 267, 133, 340
- **Economic Indicators**: Features 181, 162, 203
- **Volatility**: Features 162, 203, 133
- **Market Sentiment**: Features 340, 267, 162
- **Financial News**: Features 332, 214, 105
- **Risk Assessment**: Features 203, 162, 340
- **Portfolio Management**: Features 133, 203, 267

### Analysis Capabilities
1. **Individual Text Analysis**: Analyze single financial texts
2. **Feature Extraction**: Extract SAE features from headlines
3. **Trading Dataset Creation**: Build feature matrices for trading
4. **Model Training**: Train Random Forest models for trading
5. **Sentiment Analysis**: Analyze market sentiment across texts

## Model Path

The scripts expect the SAE model to be located at:
```
/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun
```

## Example Output

```
Starting Integrated Llama-2-7b-hf SAE Trading Analysis
Loaded SAE weights from /path/to/sae.safetensors
Step 1: Analyzing headlines for top features...
Found 45 unique features
Top 10 features by total activation:
  earnings_reports_332: 15.2341
  revenue_metrics_214: 12.8765
  stock_performance_66: 11.4321
  ...
Step 2: Building feature matrix...
Created feature matrix with 10 samples and 12 features
Step 3: Training trading model...
Model accuracy: 0.8000
Top 10 most important features:
  feature_332: 0.2341
  feature_214: 0.1876
  ...
Classification Report:
Precision: 0.7500
Recall: 0.8000
F1-Score: 0.7742
Step 4: Analyzing market sentiment...
Analyzed 10 texts
Total activations: 450
Total activation: 1234.5678
Sentiment by category:
  earnings_reports: 234.5678 total, 15.4321 avg
  revenue_metrics: 187.6543 total, 12.3456 avg
  ...
Integrated trading analysis completed successfully!
```

## Integration with Existing Trading Infrastructure

The scripts are designed to work with the existing trading infrastructure in this directory:
- Compatible with `sae_processor.py`
- Can be integrated with `main_trading_script.py`
- Follows the same patterns as other trading modules

## Troubleshooting

1. **SAE Model Not Found**: Ensure the model path is correct and the model files exist
2. **CUDA Out of Memory**: Reduce batch size or use CPU
3. **No Activations Found**: Check if the text is properly tokenized and processed

## Notes

- The scripts use layer 16 of the Llama-2-7b-hf model
- SAE features are extracted using the pre-trained weights
- Feature mappings are based on financial concept analysis
- The approach follows the notebook's methodology for trading analysis










