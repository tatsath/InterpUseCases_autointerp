#!/bin/bash

# Single-Layer AutoInterp Analysis with OpenRouter API
# This script uses OpenRouter API for better explanations and processes features in a consolidated run

LAYER=10
echo "🚀 Running AutoInterp analysis for Layer $LAYER with OpenRouter API..."

# Configuration
BASE_MODEL="meta-llama/Llama-2-7b-hf"
SAE_MODEL="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
EXPLAINER_MODEL="openai/gpt-4o-mini"
N_TOKENS=10000

# Activate conda environment for SAE
echo "🐍 Activating conda environment: sae"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae

# Set environment variables for better performance
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1

# OpenRouter API key should be set as environment variable
# export OPENROUTER_API_KEY="your_key_here"

# Get top 10 features for comprehensive analysis
FEATURES=$(python3 -c "
import pandas as pd
df = pd.read_csv('multi_layer_lite_results/features_layer${LAYER}.csv')
features = df['feature'].head(10).tolist() if 'feature' in df.columns else df.iloc[:10, 1].tolist()
print(' '.join(map(str, features)))
")

echo "🎯 Features: $FEATURES"
echo "🔧 Using OpenRouter API for better explanations"

# Process all features in a single run to avoid separate folders
echo "🔍 Processing all features in a single run..."
cd ../../autointerp/autointerp_full
RUN_NAME="single_layer_openrouter_layer${LAYER}_temp"

# Convert features array to space-separated list for --feature_num
FEATURE_LIST=$(echo $FEATURES | tr ' ' ' ')

python -m autointerp_full \
    "$BASE_MODEL" \
    "$SAE_MODEL" \
    --n_tokens "$N_TOKENS" \
    --feature_num $FEATURE_LIST \
    --hookpoints "layers.$LAYER" \
    --scorers detection \
    --explainer_model "$EXPLAINER_MODEL" \
    --explainer_provider "openrouter" \
    --explainer_model_max_len 512 \
    --num_gpus 4 \
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

# Move results to FinanceLabeling directory
if [ -d "results/$RUN_NAME" ]; then
    echo "📋 Moving results to FinanceLabeling directory..."
    # Remove existing results directory if it exists
    rm -rf "../../InterpUseCases_autointerp/FinanceLabeling/single_layer_openrouter_results"
    # Move the entire results directory
    mv "results/$RUN_NAME" "../../InterpUseCases_autointerp/FinanceLabeling/single_layer_openrouter_results"
    
    # Generate CSV summary
    if [ -d "../../InterpUseCases_autointerp/FinanceLabeling/single_layer_openrouter_results/explanations" ] && [ "$(ls -A ../../InterpUseCases_autointerp/FinanceLabeling/single_layer_openrouter_results/explanations)" ]; then
        echo "📊 Generating CSV summary..."
        python generate_results_csv.py "../../InterpUseCases_autointerp/FinanceLabeling/single_layer_openrouter_results"
        echo "✅ CSV summary generated in FinanceLabeling directory"
    fi
fi

# Go back to FinanceLabeling directory
cd ../../InterpUseCases_autointerp/FinanceLabeling
echo "✅ Completed consolidated analysis for all features"

echo "🎉 All features processed successfully with OpenRouter API!"
echo "📁 Results saved in: single_layer_openrouter_results/"
echo "📊 CSV summary generated for all features"

# Clean up unnecessary directories to save space
echo "🧹 Cleaning up unnecessary directories to save space..."
python3 -c "
import os
import shutil

# Clean up latents and log directories to save space
latents_dir = 'single_layer_openrouter_results/latents'
log_dir = 'single_layer_openrouter_results/log'
if os.path.exists(latents_dir):
    shutil.rmtree(latents_dir)
    print('🗑️  Removed latents directory')
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
    print('🗑️  Removed log directory')
print('✅ Cleanup completed')
"

# Clean up any remaining temporary files
echo "🧹 Cleaning up temporary files..."
rm -rf "../../autointerp/autointerp_full/results/single_layer_openrouter_layer${LAYER}_temp"
