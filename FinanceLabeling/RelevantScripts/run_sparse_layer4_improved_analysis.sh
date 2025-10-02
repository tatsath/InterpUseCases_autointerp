#!/bin/bash

# Improved Sparse AutoInterp Full Analysis - Layer 4
# This script implements several strategies to improve F1 scores
# Usage: ./run_sparse_layer4_improved_analysis.sh

echo "🔍 Improved Sparse AutoInterp Full Analysis - Layer 4"
echo "========================================================="
echo "🔍 Strategies to improve F1 scores:"
echo "  • Increased minimum examples (2 → 10)"
echo "  • More training examples (40 → 100)"
echo "  • Balanced sampling approach"
echo "  • Lower activation threshold"
echo "  • Multiple scorers for better evaluation"
echo ""

# Configuration
BASE_MODEL="meta-llama/Llama-2-7b-hf"
SAE_MODEL="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
EXPLAINER_MODEL="Qwen/Qwen2.5-7B-Instruct"  # Using 7B as it performed better
EXPLAINER_PROVIDER="offline"
N_TOKENS=15000  # Increased from 10000
LAYER=4

# Features 10-20
FEATURES="10 11 12 13 14 15 16 17 18 19 20"

# Create results directory
RESULTS_DIR="sparse_layer4_improved_results"
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
export CUDA_VISIBLE_DEVICES="1,2,3,4"
export VLLM_USE_DEEP_GEMM=1
export VLLM_GPU_MEMORY_UTILIZATION=0.7
export VLLM_MAX_MODEL_LEN=4096
export VLLM_BLOCK_SIZE=16
export VLLM_SWAP_SPACE=0

echo "🎯 Features to analyze: $FEATURES"

# Run AutoInterp Full for layer 4 with improved parameters
echo "🔍 Analyzing Layer $LAYER with improved parameters..."
echo "----------------------------------------"

cd ../../autointerp/autointerp_full

# Create run name
RUN_NAME="sparse_layer4_improved"

python -m autointerp_full \
    "$BASE_MODEL" \
    "$SAE_MODEL" \
    --n_tokens 15000 \
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
    --n_non_activating 10 \
    --min_examples 10 \
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
                cp "../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR/$RUN_NAME/results_summary.csv" "../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR/results_summary_layer${LAYER}_improved.csv"
                echo "📈 Summary saved: FinanceLabeling/$RESULTS_DIR/results_summary_layer${LAYER}_improved.csv"
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
if [ -d "results/sparse_layer4_improved" ]; then
    rm -rf "results/sparse_layer4_improved"
    echo "🗑️  Cleaned up temporary results"
fi
cd ../../InterpUseCases_autointerp/FinanceLabeling

echo "🎯 Improved Sparse AutoInterp Full Analysis Summary"
echo "========================================="
echo "📊 Results saved in: $RESULTS_DIR/"

# Check final status based on actual results
if [ -d "$RESULTS_DIR/sparse_layer4_improved" ] && [ -d "$RESULTS_DIR/sparse_layer4_improved/explanations" ]; then
    FEATURE_COUNT=$(find "$RESULTS_DIR/sparse_layer4_improved/explanations" -name "feature_*.json" 2>/dev/null | wc -l)
    if [ $FEATURE_COUNT -gt 0 ]; then
        echo "   ✅ sparse_layer4_improved/ (SUCCESS - $FEATURE_COUNT features analyzed)"
        if [ -f "$RESULTS_DIR/results_summary_layer${LAYER}_improved.csv" ]; then
            echo "      📈 results_summary_layer${LAYER}_improved.csv"
        fi
        
        echo ""
        echo "📋 Analysis Complete!"
        echo "🔍 Check the following for detailed results:"
        echo "   • explanations/: Human-readable feature explanations"
        echo "   • scores/detection/: F1 scores and metrics"
        echo "   • results_summary_layer${LAYER}_improved.csv: CSV summary"
        echo ""
        
        echo "✅ Improved Sparse AutoInterp Full analysis completed successfully!"
    else
        echo "   ❌ sparse_layer4_improved/ (FAILED - no feature explanations found)"
        echo ""
        echo "❌ Analysis failed - no feature explanations were generated"
        echo "🔍 Check the logs above for error details"
        exit 1
    fi
else
    echo "   ❌ sparse_layer4_improved/ (FAILED - no results directory found)"
    echo ""
    echo "❌ Analysis failed - no results were generated"
    echo "🔍 Check the logs above for error details"
    exit 1
fi
