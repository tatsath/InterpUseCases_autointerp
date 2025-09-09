#!/bin/bash

# Final Finetuning Impact Analysis Script
# Compares SAE weights between base and finetuned models
# Usage: ./run_finetuning_analysis.sh

echo "🚀 TOP 10 FEATURES PER LAYER - FINETUNING IMPACT"
echo "================================================"
echo "📋 Each layer has 400 INDEPENDENT features (0-399)"
echo "📋 Feature 205 in Layer 4 ≠ Feature 205 in Layer 10"
echo "📊 Analyzing activation changes on financial data (functional impact of finetuning)"
echo ""
echo "🔍 Showing top 10 features with largest activation improvements per layer:"
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
if [[ "$CONDA_DEFAULT_ENV" != "sae" ]]; then
    echo "❌ Failed to activate conda environment 'sae'"
    exit 1
fi

echo "✅ Conda environment 'sae' activated"
echo ""

# Set paths
BASE_SAE="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
FINETUNED_SAE="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_finance_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
BASE_MODEL="meta-llama/Llama-2-7b-hf"
FINETUNED_MODEL="cxllin/Llama2-7b-Finance"

# Check if files exist
echo "🔍 Checking if required files exist..."

if [ ! -d "$BASE_SAE" ]; then
    echo "❌ Base SAE model not found: $BASE_SAE"
    exit 1
fi

if [ ! -d "$FINETUNED_SAE" ]; then
    echo "❌ Finetuned SAE model not found: $FINETUNED_SAE"
    exit 1
fi

echo "✅ All required files found"
echo ""

# Run the final analysis
echo "🚀 Running final finetuning impact analysis..."
echo ""

python analyze_finetuning_impact.py \
    --base_sae "$BASE_SAE" \
    --finetuned_sae "$FINETUNED_SAE" \
    --base_model "$BASE_MODEL" \
    --finetuned_model "$FINETUNED_MODEL" \
    --layers 4 10 16 22 28

echo ""
echo "✅ Top 10 features analysis complete!"
echo "📊 Results saved to: finetuning_impact_results.json"
echo ""
echo "🎯 This analysis shows:"
echo "  • Top 10 features with largest activation improvements per layer"
echo "  • Top 10 most activated features in finetuned model per layer"
echo "  • Real activations on financial data from Yahoo Finance dataset"
echo "  • Each layer's 400 features analyzed independently"