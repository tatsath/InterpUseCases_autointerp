# Feature Activation Analysis Tools

## 🔍 Overview
Two Streamlit apps for analyzing feature activations in Llama-2-7B using Sparse Autoencoders (SAEs). Understand which internal features activate when processing text.

## 🚀 Quick Start

### Text Tracing App (Direct Text Analysis)
```bash
conda activate sae
streamlit run Text_Tracing_app.py --server.port 8501
```

### Reply Tracing App (Question-Answer Analysis)
```bash
conda activate sae
streamlit run Reply_Tracing_app.py --server.port 8502
```

## 📱 Applications

### 1. Text Tracing App
**Purpose**: Analyze feature activations in any input text.
- Direct text analysis with layer-by-layer feature visualization
- Smart filtering (removes function words, focuses on domain terms)
- Token-level analysis showing which features activate per word

### 2. Reply Tracing App
**Purpose**: Question-answer analysis with feature activation insights.
- Ask questions → Llama generates responses → analyze response features
- Shows both question and generated answer
- Understands model's internal reasoning process

## 🎯 Sample Prompts

### Text Tracing App
```
"The Federal Reserve raised interest rates by 0.25% to combat inflation, causing stock markets to decline as investors reassessed growth prospects."

"Analyze how the European Central Bank's new policy stance is impacting European bank earnings, citing leadership comments from Deutsche Bank, HSBC, and BNP Paribas."
```

### Reply Tracing App
```
"What are the key factors driving inflation in the current economy?"

"How do interest rates affect stock market performance?"

"Evaluate the risks and opportunities in the fintech sector after recent SEC regulatory proposals."
```

## 🎯 Key Features

- **Layer-by-Layer Analysis**: Track features across SAE layers (4, 10, 16, 22, 28)
- **Smart Filtering**: Removes function words, focuses on domain-specific terms
- **Interactive Visualizations**: Bar charts, heatmaps, layer comparisons
- **Token-Level Analysis**: See which features activate on each word
- **Multiple Modes**: Combined, Positive/Negative Split, Token-Level Analysis
- **Always-On Filtering**: Removes features that activate regardless of content

## 🔧 Technical Details

- **Model**: `meta-llama/Llama-2-7b-hf` (Base Llama-2-7B)
- **SAE**: `llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun`
- **Layers**: 4, 10, 16, 22, 28 | **Features per Layer**: 400
- **Visualization**: Interactive Plotly charts

## 📁 Files

- **`Text_Tracing_app.py`** - Direct text analysis
- **`Reply_Tracing_app.py`** - Question-answer analysis
- **`SAE_Training_Metrics_Guide.md`** - SAE training metrics guide

## 🎯 Usage Tips

1. **Start with financial text** for best results
2. **Try different prompts** to see feature activation patterns
3. **Use Token-Level Analysis** to see which words trigger which features
4. **Compare results** across different content types
