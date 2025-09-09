#!/bin/bash

# Feature Labeling Script for Base and Finetuned Models
# This script generates labels for top features from both models
# Usage: ./run_feature_labeling.sh

echo "🏷️ FEATURE LABELING FOR BASE AND FINETUNED MODELS"
echo "=================================================="
echo "📋 Each layer has 400 INDEPENDENT features (0-399)"
echo "📋 Feature 205 in Layer 4 ≠ Feature 205 in Layer 10"
echo "📊 Generating labels for top features from both models"
echo ""
echo "🔍 Showing top 10 features with labels per model per layer:"
echo "  • Base Llama Model: meta-llama/Llama-2-7b-hf"
echo "  • Finetuned Llama Model: cxllin/Llama2-7b-Finance"
echo "  • Financial Data: jyanimaulik/yahoo_finance_stockmarket_news"
echo "  • Layers: 4, 10, 16, 22, 28"
echo "  • Each layer analyzed independently"
echo ""

# Activate conda environment
echo "🔧 Activating conda environment 'sae'..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae

# Check if environment is activated
if [ "$CONDA_DEFAULT_ENV" != "sae" ]; then
    echo "❌ Failed to activate conda environment 'sae'"
    exit 1
fi

echo "✅ Conda environment 'sae' activated"
echo ""

# Run feature labeling
echo "🚀 Starting feature labeling analysis..."
echo ""

python label_features.py \
    --base_sae "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun" \
    --finetuned_sae "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_finance_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun" \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --finetuned_model "cxllin/Llama2-7b-Finance" \
    --layers 4 10 16 22 28 \
    --top_n 10 \
    --max_samples 100

# Check if analysis was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Feature labeling analysis completed successfully!"
    echo "📊 Results saved to: feature_labels_results.json"
    echo ""
    echo "🎯 This analysis shows:"
    echo "  • Top 10 features for base model per layer"
    echo "  • Top 10 features for finetuned model per layer"
    echo "  • Labels for each feature (placeholder for now)"
    echo "  • Each layer's 400 features analyzed independently"
else
    echo ""
    echo "❌ Feature labeling analysis failed"
    exit 1
fi

echo ""
echo "✅ Feature labeling analysis complete!"
