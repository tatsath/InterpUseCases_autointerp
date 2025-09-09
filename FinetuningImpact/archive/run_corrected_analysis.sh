#!/bin/bash

# Corrected SAE Analysis Script
# Properly analyzes SAE weights understanding that features are independent across layers

echo "🚀 Corrected SAE Weight Comparison Analysis"
echo "==========================================="
echo "📋 IMPORTANT: Each layer has 400 independent features (0-399)"
echo "📋 Feature 205 in Layer 4 ≠ Feature 205 in Layer 10"
echo "📋 We analyze patterns across layers, not feature correspondence"
echo ""
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

# Run the corrected analysis
echo "🚀 Running corrected SAE analysis..."
echo ""

python corrected_sae_analysis.py \
    --base_sae "$BASE_SAE" \
    --finetuned_sae "$FINETUNED_SAE" \
    --layers 4 10 16 22 28

echo ""
echo "✅ Corrected SAE analysis complete!"
echo "📊 Results saved to: corrected_sae_analysis_results.json"
echo ""
echo "🎯 This corrected analysis shows:"
echo "  • Each layer analyzed independently (400 features per layer)"
echo "  • Cross-layer patterns and trends"
echo "  • Statistical summary across all layers"
echo "  • Proper understanding that features don't correspond across layers"
