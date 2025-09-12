#!/bin/bash

# AutoInterp Analysis for Base Model (meta-llama/Llama-2-7b-hf) - All Layers
# Optimized parameters for better F1 scores and proper financial labels

set -e

echo "🔍 Running OPTIMIZED AutoInterp analysis for BASE model - ALL LAYERS"
echo "📊 Model: meta-llama/Llama-2-7b-hf"
echo "🎯 Features: Top improved features from finetuning analysis (All Layers)"
echo "📚 Dataset: Financial data (Yahoo Finance) - OPTIMIZED PARAMETERS"
echo ""

# Set environment
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# OpenRouter API Key
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "⚠️  Warning: OPENROUTER_API_KEY environment variable not set"
    echo "   Please set it with: export OPENROUTER_API_KEY='your-api-key'"
    echo "   Or the script will use the default OpenRouter configuration"
fi

# Model paths
BASE_MODEL="meta-llama/Llama-2-7b-hf"
BASE_SAE_PATH="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"

# Output directory
OUTPUT_DIR="autointerp_base_model_all_layers_results"
mkdir -p "$OUTPUT_DIR"

# Layers to analyze
LAYERS=(4 10 16 22 28)

# Run analysis for each layer
for layer in "${LAYERS[@]}"; do
    echo "🔍 Analyzing Layer $layer..."
    echo "📚 Using financial dataset with optimized AutoInterp settings"
    echo ""
    
    # Check if results already exist for this layer
    RUN_NAME="base_model_layer${layer}_all_layers"
    if [ -d "/home/nvidia/Documents/Hariom/autointerp/autointerp_full/results/$RUN_NAME" ]; then
        echo "⏭️  Layer $layer results already exist, skipping..."
        echo "📁 Existing results: /home/nvidia/Documents/Hariom/autointerp/autointerp_full/results/$RUN_NAME"
        echo ""
        continue
    fi
    
    # Get top improved features for this layer from our results
    FEATURES=$(python3 -c "
import json
try:
    with open('finetuning_impact_results.json', 'r') as f:
        data = json.load(f)
    features = data['$layer']['top_10_improved_features']['feature_indices'][:10]
    print(' '.join(map(str, features)))
except Exception as e:
    print('Error:', e)
    exit(1)
")
    
    if [ -z "$FEATURES" ]; then
        echo "❌ Failed to extract features for layer $layer"
        echo "⏭️  Skipping layer $layer and continuing to next layer..."
        continue
    fi
    
    echo "🎯 Features: $FEATURES"
    
    # Run AutoInterp Full for this layer
    cd /home/nvidia/Documents/Hariom/autointerp/autointerp_full
    
    # Create run name for optimized analysis
    RUN_NAME="base_model_layer${layer}_all_layers"
    
    python -m autointerp_full \
        "$BASE_MODEL" \
        "$BASE_SAE_PATH" \
        --n_tokens 20000 \
        --feature_num $FEATURES \
        --hookpoints "layers.$layer" \
        --scorers detection \
        --explainer_model "meta-llama/llama-3.1-8b-instruct" \
        --explainer_provider "openrouter" \
        --explainer_model_max_len 4096 \
        --num_gpus 1 \
        --num_examples_per_scorer_prompt 2 \
        --n_non_activating 5 \
        --min_examples 3 \
        --non_activating_source "FAISS" \
        --faiss_embedding_model "sentence-transformers/all-MiniLM-L6-v2" \
        --faiss_embedding_cache_dir ".embedding_cache" \
        --faiss_embedding_cache_enabled \
        --dataset_repo "jyanimaulik/yahoo_finance_stockmarket_news" \
        --dataset_name "default" \
        --dataset_split "train[:20%]" \
        --filter_bos \
        --verbose \
        --name "$RUN_NAME" || {
        echo "❌ Error in layer $layer analysis, continuing to next layer..."
        continue
    }
    
    echo "✅ Layer $layer completed with OPTIMIZED parameters"
    echo "📁 Results saved to: /home/nvidia/Documents/Hariom/autointerp/autointerp_full/results/$RUN_NAME"
    echo ""
    echo "🎯 Analysis Summary:"
    echo "  • Model: $BASE_MODEL"
    echo "  • Layer: $layer"
    echo "  • Dataset: Yahoo Finance (financial text) - OPTIMIZED"
    echo "  • Features: $FEATURES"
    echo "  • Run Name: $RUN_NAME"
    echo "  • Optimizations: More tokens, better examples, improved contrastive learning"
    echo ""
    
    # Return to original directory for next iteration
    cd /home/nvidia/Documents/Hariom/InterpUseCases_autointerp/FinetuningImpact
done

echo "✅ All layers completed for BASE model"
echo "📁 All results saved to: /home/nvidia/Documents/Hariom/autointerp/autointerp_full/results/"
