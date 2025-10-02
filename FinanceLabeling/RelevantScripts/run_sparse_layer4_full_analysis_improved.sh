#!/bin/bash

# Sparse AutoInterp Full Analysis - Layer 4 (IMPROVED VERSION)
# This script runs AutoInterp Full analysis with optimized parameters for better F1 scores
# Usage: ./run_sparse_layer4_full_analysis_improved.sh

echo "🔍 Sparse AutoInterp Full Analysis - Layer 4 (IMPROVED)"
echo "========================================================"
echo "🔍 Running optimized analysis on features 10-20 of layer 4:"
echo "  • Layer: 4"
echo "  • Features: 10-20 (features 10 through 20)"
echo "  • Domain: General (sparse features)"
echo "  • LLM-based explanations with confidence scores"
echo "  • F1 scores, precision, and recall metrics"
echo "  • OPTIMIZED for better F1 scores"
echo ""

# Configuration
BASE_MODEL="meta-llama/Llama-2-7b-hf"
SAE_MODEL="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
EXPLAINER_MODEL="Qwen/Qwen2.5-3B-Instruct"
EXPLAINER_PROVIDER="offline"
N_TOKENS=10000  # Increased for better examples
LAYER=4

# Features 10-20
FEATURES="10 11 12 13 14 15 16 17 18 19 20"

# Create results directory
RESULTS_DIR="sparse_layer4_full_results_improved"
mkdir -p "$RESULTS_DIR"

echo "📁 Results will be saved to: $RESULTS_DIR/"
echo ""

# Activate conda environment for SAE
echo "🐍 Activating conda environment: sae"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae
echo ""

# Set environment variables for better performance (optimized for F1 scores)
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="1,2,3,4"  # Use 4 free GPUs (excluding GPU 0) - 16 heads / 4 GPUs = 4 heads per GPU
export VLLM_USE_DEEP_GEMM=1  # Enable deep GEMM like vllm_serve.sh
# Remove NCCL disables to allow proper GPU communication
export VLLM_GPU_MEMORY_UTILIZATION=0.7  # Increased to match working vllm_serve.sh
export VLLM_MAX_MODEL_LEN=4096  # Increased to handle longer prompts
export VLLM_BLOCK_SIZE=16
export VLLM_SWAP_SPACE=0

# Create run name with timestamp
RUN_NAME="sparse_layer4_full_improved_$(date +%Y%m%d_%H%M%S)"

echo "🎯 Features to analyze: $FEATURES"
echo "🔍 Analyzing Layer 4 with AutoInterp Full (IMPROVED)..."
echo "----------------------------------------"

# Run AutoInterp Full with IMPROVED parameters for better F1 scores
cd /home/nvidia/Documents/Hariom/autointerp/autointerp_full
python -m autointerp_full \
    "$BASE_MODEL" \
    "$SAE_MODEL" \
    --n_tokens $N_TOKENS \
    --cache_ctx_len 1024 \
    --batch_size 16 \
    --feature_num $FEATURES \
    --hookpoints "layers.$LAYER" \
    --scorers detection \
    --explainer_model "$EXPLAINER_MODEL" \
    --explainer_provider "$EXPLAINER_PROVIDER" \
    --explainer_model_max_len 4096 \
    --num_gpus 4 \
    --num_examples_per_scorer_prompt 3 \
    --n_non_activating 10 \
    --min_examples 5 \
    --non_activating_source "FAISS" \
    --faiss_embedding_model "sentence-transformers/all-MiniLM-L6-v2" \
    --faiss_embedding_cache_dir ".embedding_cache" \
    --faiss_embedding_cache_enabled \
    --dataset_repo wikitext \
    --dataset_name wikitext-103-raw-v1 \
    --dataset_split "train[:10%]" \
    --filter_bos \
    --verbose \
    --name "$RUN_NAME"

# Check if analysis was successful
if [ -d "$RESULTS_DIR/$RUN_NAME" ]; then
    echo "✅ Analysis completed successfully!"
    echo "📋 Results moved to: FinanceLabeling/$RESULTS_DIR/$RUN_NAME/"
    
    # Generate CSV summary
    echo "📊 Generating CSV summary for layer 4..."
    python -c "
import pandas as pd
import json
import os
from pathlib import Path

# Find the results directory
results_dir = Path('$RESULTS_DIR/$RUN_NAME')
if results_dir.exists():
    # Load results
    results_file = results_dir / 'results_summary.csv'
    if results_file.exists():
        df = pd.read_csv(results_file)
        
        # Sort by F1 score descending
        df_sorted = df.sort_values('f1_score', ascending=False)
        
        # Save sorted results
        output_file = Path('$RESULTS_DIR/results_summary_layer4_improved.csv')
        df_sorted.to_csv(output_file, index=False)
        
        print(f'✅ CSV summary generated: {output_file}')
        
        # Display top features
        print(f'📊 Found {len(df)} features with results')
        for _, row in df_sorted.head(10).iterrows():
            print(f'  Feature {row[\"feature\"]}: {row[\"label\"]} (F1: {row[\"f1_score\"]:.3f})')
        
        # Save summary
        summary_file = Path('FinanceLabeling/$RESULTS_DIR/results_summary_layer4_improved.csv')
        df_sorted.to_csv(summary_file, index=False)
        print(f'📈 Summary saved: {summary_file}')
    else:
        print('❌ No results_summary.csv found')
else
    echo "❌ Layer 4 AutoInterp Full analysis failed - no results directory found"
fi

# Clean up temporary files
echo ""
echo "🧹 Cleaning up temporary files..."

# Display final summary
echo ""
echo "🎯 Sparse AutoInterp Full Analysis Summary - IMPROVED"
echo "====================================================="
echo "📊 Results saved in: $RESULTS_DIR/"
if [ -d "$RESULTS_DIR/$RUN_NAME" ]; then
    echo "   ✅ $RUN_NAME/ - SUCCESS"
else
    echo "   ❌ $RUN_NAME/ - FAILED - no results directory found"
fi

if [ -d "$RESULTS_DIR/$RUN_NAME" ]; then
    echo ""
    echo "✅ Analysis completed successfully!"
    echo "🔍 Check the results in: $RESULTS_DIR/$RUN_NAME/"
    echo "📊 CSV summary: $RESULTS_DIR/results_summary_layer4_improved.csv"
else
    echo ""
    echo "❌ Analysis failed - check the logs above for error details"
fi
