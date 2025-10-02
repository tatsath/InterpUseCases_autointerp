#!/bin/bash

# Sparse AutoInterp Lite Analysis - Layer 4
# This script runs AutoInterp Lite analysis on layer 4 for sparse autoencoder features
# Usage: ./run_sparse_layer4_lite_analysis.sh

echo "🔍 Sparse AutoInterp Lite Analysis - Layer 4"
echo "============================================="
echo "🔍 Running feature discovery on layer 4 for sparse autoencoder:"
echo "  • Layer: 4"
echo "  • Domain: General (sparse features)"
echo "  • Finding top sparse autoencoder features"
echo "  • Activation strength and specialization scores"
echo ""

# Configuration
BASE_MODEL="meta-llama/Llama-2-7b-hf"
SAE_MODEL="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
LAYER=4
N_TOKENS=20000

# Create results directory
RESULTS_DIR="sparse_layer4_lite_results"
mkdir -p "$RESULTS_DIR"

echo "📁 Results will be saved to: $RESULTS_DIR/"
echo ""

# Activate conda environment for SAE
echo "🐍 Activating conda environment: sae"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae
echo ""

# Set environment variables for better performance
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1

# Run AutoInterp Lite for sparse features
echo "🔍 Running AutoInterp Lite for sparse features on layer $LAYER..."
cd ../../autointerp/autointerp_lite

python run_analysis.py \
    --mode general \
    --base_model "$BASE_MODEL" \
    --sae_model "$SAE_MODEL" \
    --layer "$LAYER" \
    --n_tokens "$N_TOKENS" \
    --output_dir "../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR"

# Check if analysis was successful
if [ $? -eq 0 ]; then
    echo "✅ Sparse AutoInterp Lite analysis completed successfully"
    
    # Check if results were created
    if [ -f "../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR/features_layer${LAYER}.csv" ]; then
        echo "📊 Results saved: $RESULTS_DIR/features_layer${LAYER}.csv"
        
        # Show top 10 features
        echo "🏆 Top 10 sparse features found:"
        python3 -c "
import pandas as pd
try:
    df = pd.read_csv('../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR/features_layer${LAYER}.csv')
    print(df.head(10).to_string(index=False))
except Exception as e:
    print('Error reading results:', e)
"
    else
        echo "❌ Results file not found"
    fi
else
    echo "❌ Sparse AutoInterp Lite analysis failed"
fi

cd ../../InterpUseCases_autointerp/FinanceLabeling

echo ""
echo "🎯 Sparse AutoInterp Lite Analysis Summary"
echo "=========================================="
echo "📊 Results saved in: $RESULTS_DIR/"
if [ -f "$RESULTS_DIR/features_layer${LAYER}.csv" ]; then
    echo "   ✅ features_layer${LAYER}.csv"
    echo "   📈 Top sparse autoencoder features identified"
else
    echo "   ❌ No results found"
fi

echo ""
echo "✅ Sparse AutoInterp Lite analysis completed!"
echo "🔍 Next step: Run ./run_sparse_layer4_full_analysis.sh for detailed explanations"
