#!/bin/bash

# AutoInterp Analysis for Finetuned Model (cxllin/Llama2-7b-Finance)
# Runs analysis on top activated features from all layers

set -e

echo "🔍 Running AutoInterp analysis for FINETUNED model"
echo "📊 Model: cxllin/Llama2-7b-Finance"
echo "🎯 Features: Top improved features from finetuning analysis (SAME as base model)"
echo ""

# Set environment
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# OpenRouter API Key (set this environment variable)
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "⚠️  Warning: OPENROUTER_API_KEY environment variable not set"
    echo "   Please set it with: export OPENROUTER_API_KEY='your-api-key'"
    echo "   Or the script will use the default OpenRouter configuration"
fi

# Model paths
FINETUNED_MODEL="cxllin/Llama2-7b-Finance"
FINETUNED_SAE_PATH="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_finance_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"

# Output directory
OUTPUT_DIR="autointerp_finetuned_model_results"
mkdir -p "$OUTPUT_DIR"

# Layers to analyze
LAYERS=(4 10 16 22 28)

# Run analysis for each layer
for layer in "${LAYERS[@]}"; do
    echo "🔍 Analyzing Layer $layer..."
    
    # Get top improved features for this layer from our results (SAME as base model)
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
        continue
    fi
    
    echo "🎯 Features: $FEATURES"
    
    # Run AutoInterp Full for this layer
    cd /home/nvidia/Documents/Hariom/autointerp/autointerp_full
    
    # Create layer-specific run name
    RUN_NAME="finetuned_model_layer${layer}"
    
    python -m autointerp_full \
        "$FINETUNED_MODEL" \
        "$FINETUNED_SAE_PATH" \
        --n_tokens 10000 \
        --feature_num $FEATURES \
        --hookpoints "layers.$layer" \
        --scorers detection \
        --explainer_model "meta-llama/llama-3.1-8b-instruct" \
        --explainer_provider "openrouter" \
        --explainer_model_max_len 4096 \
        --num_gpus 1 \
        --num_examples_per_scorer_prompt 1 \
        --n_non_activating 2 \
        --min_examples 1 \
        --non_activating_source "FAISS" \
        --faiss_embedding_model "sentence-transformers/all-MiniLM-L6-v2" \
        --faiss_embedding_cache_dir ".embedding_cache" \
        --faiss_embedding_cache_enabled \
        --dataset_repo jyanimaulik/yahoo_finance_stockmarket_news \
        --dataset_name default \
        --dataset_split "train[:1%]" \
        --filter_bos \
        --verbose \
        --name "$RUN_NAME"
    
    echo "✅ Layer $layer completed"
    echo ""
done

echo "✅ AutoInterp analysis completed for FINETUNED model"
echo "📁 Results saved to: /home/nvidia/Documents/Hariom/autointerp/autointerp_full/results/"