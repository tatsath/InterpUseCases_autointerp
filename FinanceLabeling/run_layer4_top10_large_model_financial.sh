#!/bin/bash

# Layer 4 Top 10 Features Analysis with Large Model + Financial Dataset
# This script runs AutoInterp Full analysis on the top 10 features from layer 4
# using a larger, more capable model with the same financial dataset as previous analysis
# Usage: ./run_layer4_top10_large_model_financial.sh

echo "🚀 Layer 4 Top 10 Features Analysis - Large Model + Financial Dataset"
echo "====================================================================="
echo "🔍 Running detailed analysis on top 10 features from layer 4:"
echo "  • Layer: 4"
echo "  • Top 10 features (from previous analysis)"
echo "  • Large model (72B) for better consistency"
echo "  • Financial dataset (Yahoo Finance) for domain consistency"
echo "  • Improved parameters for higher F1 scores"
echo "  • LLM-based explanations with confidence scores"
echo ""

# Configuration
BASE_MODEL="meta-llama/Llama-2-7b-hf"
SAE_MODEL="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
EXPLAINER_MODEL="Qwen/Qwen2.5-72B-Instruct"  # Using the larger 72B model
EXPLAINER_PROVIDER="offline"
N_TOKENS=50000  # Increased for more data
LAYER=4

# Features 1 to 10 for analysis
FEATURES="1 2 3 4 5 6 7 8 9 10"

# Create results directory
RESULTS_DIR="Test_results_improved"
mkdir -p "$RESULTS_DIR"

echo "📁 Results will be saved to: $RESULTS_DIR/"
echo "🎯 Features to analyze: $FEATURES"
echo "🤖 Using large model: $EXPLAINER_MODEL"
echo "📊 Using financial dataset: jyanimaulik/yahoo_finance_stockmarket_news"
echo ""

# Activate conda environment for SAE
echo "🐍 Activating conda environment: sae"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae
echo ""

# Set environment variables for better performance (optimized for large model)
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="1,2,3,4"  # Use GPUs 1-4 to avoid processes on GPU 0
export VLLM_USE_DEEP_GEMM=1
export VLLM_GPU_MEMORY_UTILIZATION=0.6  # Reduced utilization for memory constraints
export VLLM_MAX_MODEL_LEN=8192  # Increased for longer contexts
export VLLM_BLOCK_SIZE=32
export VLLM_SWAP_SPACE=0

# Run AutoInterp Full for layer 4 with large model and financial dataset
echo "🔍 Analyzing Layer $LAYER with AutoInterp Full (Large Model + Financial Dataset)..."
echo "----------------------------------------"

cd ../../autointerp/autointerp_full

# Create run name
RUN_NAME="layer4_top10_large_model_financial"

python -m autointerp_full \
    "$BASE_MODEL" \
    "$SAE_MODEL" \
    --n_tokens "$N_TOKENS" \
    --cache_ctx_len 1024 \
    --batch_size 4 \
    --feature_num $FEATURES \
    --hookpoints "layers.$LAYER" \
    --scorers detection \
    --explainer_model "$EXPLAINER_MODEL" \
    --explainer_provider "$EXPLAINER_PROVIDER" \
    --explainer_model_max_len 8192 \
    --num_gpus 4 \
    --num_examples_per_scorer_prompt 15 \
    --n_non_activating 25 \
    --min_examples 50 \
    --non_activating_source "FAISS" \
    --faiss_embedding_model "sentence-transformers/all-MiniLM-L6-v2" \
    --faiss_embedding_cache_dir ".embedding_cache" \
    --faiss_embedding_cache_enabled \
    --dataset_repo jyanimaulik/yahoo_finance_stockmarket_news \
    --dataset_name default \
    --dataset_split "train[:50%]" \
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
                cp "../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR/$RUN_NAME/results_summary.csv" "../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR/results_summary_layer${LAYER}_large_model_financial.csv"
                echo "📈 Summary saved: FinanceLabeling/$RESULTS_DIR/results_summary_layer${LAYER}_large_model_financial.csv"
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
if [ -d "results/layer4_top10_large_model_financial" ]; then
    rm -rf "results/layer4_top10_large_model_financial"
    echo "🗑️  Cleaned up temporary results"
fi
cd ../../InterpUseCases_autointerp/FinanceLabeling

echo "🎯 Layer 4 Top 10 Features Analysis Summary (Large Model + Financial Dataset)"
echo "============================================================================="
echo "📊 Results saved in: $RESULTS_DIR/"

# Check final status based on actual results
if [ -d "$RESULTS_DIR/layer4_top10_large_model_financial" ] && [ -d "$RESULTS_DIR/layer4_top10_large_model_financial/explanations" ]; then
    FEATURE_COUNT=$(find "$RESULTS_DIR/layer4_top10_large_model_financial/explanations" -name "feature_*.json" 2>/dev/null | wc -l)
    if [ $FEATURE_COUNT -gt 0 ]; then
        echo "   ✅ layer4_top10_large_model_financial/ (SUCCESS - $FEATURE_COUNT features analyzed)"
        if [ -f "$RESULTS_DIR/results_summary_layer${LAYER}_large_model_financial.csv" ]; then
            echo "      📈 results_summary_layer${LAYER}_large_model_financial.csv"
        fi
        
        echo ""
        echo "📋 Analysis Complete!"
        echo "🔍 Check the following for detailed results:"
        echo "   • explanations/: Human-readable feature explanations"
        echo "   • scores/detection/: F1 scores and metrics"
        echo "   • results_summary_layer${LAYER}_large_model_financial.csv: CSV summary"
        echo ""
        
        # Clean up unnecessary directories to save space
        echo "🧹 Cleaning up unnecessary directories to save space..."
        python3 -c "
import os
import shutil

# Clean up latents and log directories to save space
latents_dir = 'layer4_top10_large_model_financial_results/layer4_top10_large_model_financial/latents'
log_dir = 'layer4_top10_large_model_financial_results/layer4_top10_large_model_financial/log'
if os.path.exists(latents_dir):
    shutil.rmtree(latents_dir)
    print('🗑️  Removed latents directory')
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
    print('🗑️  Removed log directory')
print('✅ Cleanup completed')
"
        
        echo ""
        echo "✅ Layer 4 Top 10 Features analysis with large model + financial dataset completed successfully!"
        echo "🎯 Using Qwen 72B model with Yahoo Finance dataset for fair comparison!"
    else
        echo "   ❌ layer4_top10_large_model_financial/ (FAILED - no feature explanations found)"
        echo ""
        echo "❌ Analysis failed - no feature explanations were generated"
        echo "🔍 Check the logs above for error details"
        exit 1
    fi
else
    echo "   ❌ layer4_top10_large_model_financial/ (FAILED - no results directory found)"
    echo ""
    echo "❌ Analysis failed - no results were generated"
    echo "🔍 Check the logs above for error details"
    exit 1
fi
