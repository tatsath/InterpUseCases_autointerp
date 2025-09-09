#!/bin/bash

# Real Feature Labeling Script using AutoInterp
# This script generates actual meaningful labels for features using AutoInterp
# Usage: ./run_real_labeling.sh

echo "🏷️ REAL FEATURE LABELING WITH AUTOINTERP"
echo "========================================="
echo "📋 Generating actual meaningful labels for features"
echo "🔍 Using AutoInterp to explain what each feature does"
echo ""
echo "🎯 This will generate real labels like:"
echo "  • 'Financial market volatility indicators'"
echo "  • 'Stock price movement patterns'"
echo "  • 'Economic sentiment analysis'"
echo "  • Instead of: 'Feature_389_Layer_22'"
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

# Run real labeling (with reduced number of features for speed)
echo "🚀 Starting real feature labeling analysis..."
echo "⚠️  Note: This will take longer as we're generating real explanations"
echo ""

python generate_real_labels.py \
    --base_sae "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun" \
    --finetuned_sae "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_finance_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun" \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --finetuned_model "cxllin/Llama2-7b-Finance" \
    --layers 4 10 16 22 28 \
    --top_n 3 \
    --output_dir "real_labels_output"

# Check if analysis was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Real feature labeling analysis completed successfully!"
    echo "📊 Results saved to: real_labels_output/real_labels_results.json"
    echo ""
    echo "🎯 This analysis shows:"
    echo "  • Real explanations for base model features"
    echo "  • Real explanations for finetuned model features"
    echo "  • Real explanations for top improved features"
    echo "  • Meaningful labels instead of placeholder names"
else
    echo ""
    echo "❌ Real feature labeling analysis failed"
    exit 1
fi

echo ""
echo "✅ Real feature labeling analysis complete!"
