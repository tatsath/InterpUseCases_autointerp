#!/bin/bash

# Sparse AutoInterp Full Analysis - Layer 4 (Qwen 7B)
# This script runs AutoInterp Full analysis on features 10-20 of layer 4 for sparse autoencoder
# Usage: ./run_sparse_layer4_full_analysis_qwen7b.sh

echo "🔍 Sparse AutoInterp Full Analysis - Layer 4 (7B Qwen)"
echo "========================================================="
echo "🔍 Running detailed analysis on features 10-20 of layer 4:"
echo "  • Layer: 4"
echo "  • Features: 10-20 (features 10 through 20)"
echo "  • Domain: General (sparse features)"
echo "  • Explainer Model: Qwen2.5-7B-Instruct (7B parameters)"
echo "  • LLM-based explanations with confidence scores"
echo "  • F1 scores, precision, and recall metrics"
echo ""

# Configuration
BASE_MODEL="meta-llama/Llama-2-7b-hf"
SAE_MODEL="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
EXPLAINER_MODEL="Qwen/Qwen2.5-7B-Instruct"
EXPLAINER_PROVIDER="offline"
N_TOKENS=5000
LAYER=4

# Features 10-20
FEATURES="10 11 12 13 14 15 16 17 18 19 20"

# Create results directory
RESULTS_DIR="sparse_layer4_full_results_7b_qwen"
mkdir -p "$RESULTS_DIR"

echo "📁 Results will be saved to: $RESULTS_DIR/"
echo ""

# Activate conda environment for SAE
echo "🐍 Activating conda environment: sae"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae
echo ""

# Set environment variables for better performance (adjusted for 7B model)
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="1,2,3,4"  # Use 4 free GPUs (excluding GPU 0)
export VLLM_USE_DEEP_GEMM=1
export VLLM_GPU_MEMORY_UTILIZATION=0.7
export VLLM_MAX_MODEL_LEN=4096
export VLLM_BLOCK_SIZE=16
export VLLM_SWAP_SPACE=0

# Check if lite results exist (optional - we're using first 10 features anyway)
LITE_RESULTS_DIR="sparse_layer4_lite_results"
if [ -d "$LITE_RESULTS_DIR" ]; then
    echo "📋 Found lite results directory: $LITE_RESULTS_DIR"
    echo "   (Using first 10 features regardless of lite results)"
else
    echo "⚠️  No lite results found, proceeding with first 10 features (0-9)"
fi

echo "🎯 Features to analyze: $FEATURES"

# Run AutoInterp Full for layer 4
echo "🔍 Analyzing Layer $LAYER with AutoInterp Full (Qwen 7B)..."
echo "----------------------------------------"

cd ../../autointerp/autointerp_full

# Create run name
RUN_NAME="sparse_layer4_full_7b_qwen"

python -m autointerp_full \
    "$BASE_MODEL" \
    "$SAE_MODEL" \
    --n_tokens 10000 \
    --cache_ctx_len 512 \
    --batch_size 8 \
    --feature_num $FEATURES \
    --hookpoints "layers.$LAYER" \
    --scorers detection \
    --explainer_model "$EXPLAINER_MODEL" \
    --explainer_provider "offline" \
    --explainer_model_max_len 4096 \
    --num_gpus 4 \
    --num_examples_per_scorer_prompt 1 \
    --n_non_activating 5 \
    --min_examples 2 \
    --non_activating_source "FAISS" \
    --faiss_embedding_model "sentence-transformers/all-MiniLM-L6-v2" \
    --faiss_embedding_cache_dir ".embedding_cache" \
    --faiss_embedding_cache_enabled \
    --dataset_repo wikitext \
    --dataset_name wikitext-103-raw-v1 \
    --dataset_split "train[:5%]" \
    --filter_bos \
    --verbose \
    --name "$RUN_NAME"

# Check if analysis was successful
ANALYSIS_EXIT_CODE=$?
if [ $ANALYSIS_EXIT_CODE -eq 0 ]; then
    # Move results to our organized directory
    if [ -d "results/$RUN_NAME" ]; then
        # Remove existing results directory if it exists
        rm -rf "../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR/$RUN_NAME"
        # Move the entire results directory
        mv "results/$RUN_NAME" "../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR/"
        echo "📋 Results moved to: FinanceLabeling/$RESULTS_DIR/$RUN_NAME/"
        
        # Generate CSV summary if results exist
        if [ -d "../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR/$RUN_NAME/explanations" ] && [ "$(ls -A ../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR/$RUN_NAME/explanations)" ]; then
            echo "📊 Generating CSV summary for layer $LAYER..."
            python generate_results_csv.py "../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR/$RUN_NAME"
            if [ -f "../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR/$RUN_NAME/results_summary.csv" ]; then
                cp "../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR/$RUN_NAME/results_summary.csv" "../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR/results_summary_layer${LAYER}_7b.csv"
                echo "📈 Summary saved: FinanceLabeling/$RESULTS_DIR/results_summary_layer${LAYER}_7b.csv"
            fi
        fi
        
        # Verify that we actually have feature explanations
        FEATURE_COUNT=$(find "../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR/$RUN_NAME/explanations" -name "feature_*.json" 2>/dev/null | wc -l)
        if [ $FEATURE_COUNT -gt 0 ]; then
            echo "✅ Layer $LAYER AutoInterp Full analysis completed successfully"
            echo "🎯 Successfully analyzed $FEATURE_COUNT features"
        else
            echo "❌ Layer $LAYER AutoInterp Full analysis failed - no feature explanations generated"
            ANALYSIS_EXIT_CODE=1
        fi
    else
        echo "❌ Layer $LAYER AutoInterp Full analysis failed - no results directory found"
        ANALYSIS_EXIT_CODE=1
    fi
else
    echo "❌ Layer $LAYER AutoInterp Full analysis failed with exit code $ANALYSIS_EXIT_CODE"
fi

echo ""
cd ../../InterpUseCases_autointerp/FinanceLabeling

# Clean up any remaining temporary files
echo "🧹 Cleaning up temporary files..."
cd ../../autointerp/autointerp_full
if [ -d "results/sparse_layer4_full_7b_qwen" ]; then
    rm -rf "results/sparse_layer4_full_7b_qwen"
    echo "🗑️  Cleaned up temporary results"
fi
cd ../../InterpUseCases_autointerp/FinanceLabeling

echo "🎯 Sparse AutoInterp Full Analysis Summary (Qwen 7B)"
echo "========================================="
echo "📊 Results saved in: $RESULTS_DIR/"

# Check final status based on actual results
if [ -d "$RESULTS_DIR/sparse_layer4_full_7b_qwen" ] && [ -d "$RESULTS_DIR/sparse_layer4_full_7b_qwen/explanations" ]; then
    FEATURE_COUNT=$(find "$RESULTS_DIR/sparse_layer4_full_7b_qwen/explanations" -name "feature_*.json" 2>/dev/null | wc -l)
    if [ $FEATURE_COUNT -gt 0 ]; then
        echo "   ✅ sparse_layer4_full_7b_qwen/ (SUCCESS - $FEATURE_COUNT features analyzed)"
        if [ -f "$RESULTS_DIR/results_summary_layer${LAYER}_7b.csv" ]; then
            echo "      📈 results_summary_layer${LAYER}_7b.csv"
        fi
        
        echo ""
        echo "📋 Analysis Complete!"
        echo "🔍 Check the following for detailed results:"
        echo "   • explanations/: Human-readable feature explanations"
        echo "   • scores/detection/: F1 scores and metrics"
        echo "   • results_summary_layer${LAYER}_7b.csv: CSV summary"
        echo ""
        
        # Clean up unnecessary directories to save space
        echo "🧹 Cleaning up unnecessary directories to save space..."
        python3 -c "
import os
import shutil

# Clean up latents and log directories to save space
latents_dir = 'sparse_layer4_full_results_7b_qwen/sparse_layer4_full_7b_qwen/latents'
log_dir = 'sparse_layer4_full_results_7b_qwen/sparse_layer4_full_7b_qwen/log'
if os.path.exists(latents_dir):
    shutil.rmtree(latents_dir)
    print('🗑️  Removed latents directory')
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
    print('🗑️  Removed log directory')
print('✅ Cleanup completed')
"
        
        echo ""
        echo "✅ Sparse AutoInterp Full analysis completed successfully!"
    else
        echo "   ❌ sparse_layer4_full_7b_qwen/ (FAILED - no feature explanations found)"
        echo ""
        echo "❌ Analysis failed - no feature explanations were generated"
        echo "🔍 Check the logs above for error details"
        exit 1
    fi
else
    echo "   ❌ sparse_layer4_full_7b_qwen/ (FAILED - no results directory found)"
    echo ""
    echo "❌ Analysis failed - no results were generated"
    echo "🔍 Check the logs above for error details"
    exit 1
fi
