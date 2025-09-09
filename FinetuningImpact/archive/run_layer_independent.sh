#!/bin/bash

# Layer-Independent SAE Analysis Script
# Analyzes each layer completely independently

echo "🚀 LAYER-INDEPENDENT SAE ANALYSIS"
echo "================================="
echo "📋 CRITICAL: Each layer has 400 INDEPENDENT features (0-399)"
echo "📋 Feature 205 in Layer 4 ≠ Feature 205 in Layer 10 ≠ Feature 205 in Layer 16"
echo "📋 We analyze each layer completely separately"
echo ""
echo "🔍 Analyzing SAE weights to identify finetuning impact:"
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

# Run the layer-independent analysis
echo "🚀 Running layer-independent SAE analysis..."
echo ""

python layer_independent_analysis.py \
    --base_sae "$BASE_SAE" \
    --finetuned_sae "$FINETUNED_SAE" \
    --layers 4 10 16 22 28

echo ""
echo "✅ Layer-independent SAE analysis complete!"
echo "📊 Results saved to: layer_independent_analysis_results.json"
echo ""
echo "🎯 This analysis shows:"
echo "  • Top 10 features for each layer independently"
echo "  • Each layer's 400 features analyzed separately"
echo "  • Clear understanding that features don't correspond across layers"
echo "  • Detailed change values and percentages for each feature"
