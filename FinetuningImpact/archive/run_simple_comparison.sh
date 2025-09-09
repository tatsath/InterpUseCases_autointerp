#!/bin/bash

# Simple SAE Feature Comparison Script
# Directly compares SAE features to identify finetuning impact

echo "🚀 Simple SAE Feature Comparison"
echo "================================"
echo "🔍 Comparing SAE features to identify finetuning impact:"
echo "  • Base Llama Model: meta-llama/Llama-2-7b-hf"
echo "  • Finetuned Llama Model: cxllin/Llama2-7b-Finance"
echo "  • Base SAE Model: llama2_7b_hf_layers4_10_16_22_28_k32_latents400_wikitext103_torchrun"
echo "  • Finetuned SAE Model: llama2_7b_finance_layers4_10_16_22_28_k32_latents400_wikitext103_torchrun"
echo "  • Layers: 4, 10, 16, 22, 28"
echo "  • Financial Data: ../../autointerp/autointerp_lite/financial_texts.txt"
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
FINANCIAL_DATA="../../autointerp/autointerp_lite/financial_texts.txt"

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

if [ ! -f "$FINANCIAL_DATA" ]; then
    echo "❌ Financial data not found: $FINANCIAL_DATA"
    exit 1
fi

echo "✅ All required files found"
echo ""

# Run the simple comparison
echo "🚀 Running simple SAE feature comparison..."
echo ""

python simple_sae_comparison.py \
    --base_sae "$BASE_SAE" \
    --finetuned_sae "$FINETUNED_SAE" \
    --base_model "$BASE_MODEL" \
    --finetuned_model "$FINETUNED_MODEL" \
    --financial_data "$FINANCIAL_DATA" \
    --layers 4 10 16 22 28 \
    --top_k 10

echo ""
echo "✅ Simple SAE comparison complete!"
echo "📊 Results saved to: simple_sae_comparison_results.json"
echo ""
echo "🎯 This analysis shows:"
echo "  • Which features are more activated in the finetuned SAE"
echo "  • The activation improvement from finetuning"
echo "  • Top features that benefit most from finetuning"
echo "  • Number of features that improved vs degraded"
