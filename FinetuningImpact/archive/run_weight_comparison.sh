#!/bin/bash

# SAE Weight Comparison Script
# Directly compares SAE weights to identify finetuning impact

echo "🚀 SAE Weight Comparison Analysis"
echo "================================="
echo "🔍 Comparing SAE weights to identify finetuning impact:"
echo "  • Base SAE Model: llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
echo "  • Finetuned SAE Model: llama2_7b_finance_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
echo "  • Layers: 4, 10, 16, 22, 28"
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

# Run the weight comparison
echo "🚀 Running SAE weight comparison..."
echo ""

python compare_sae_weights.py \
    --base_sae "$BASE_SAE" \
    --finetuned_sae "$FINETUNED_SAE" \
    --layers 4 10 16 22 28

echo ""
echo "✅ SAE weight comparison complete!"
echo "📊 Results saved to: sae_weight_comparison_results.json"
echo ""
echo "🎯 This analysis shows:"
echo "  • Which features have the largest weight changes from finetuning"
echo "  • Which features have the largest bias changes from finetuning"
echo "  • Correlation between weight and bias changes"
echo "  • Statistical summary of changes across all layers"
