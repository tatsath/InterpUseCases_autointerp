#!/bin/bash

# Single Layer AutoInterp Analysis - Qwen 72B + Financial Dataset
# =============================================================

echo "🚀 Single Layer AutoInterp Analysis - Qwen 72B + Financial Dataset"
echo "=================================================================="
echo "🔍 Running analysis on layer 4 only:"
echo "  • Base Model: 7B parameters (Llama-2-7b-hf)"
echo "  • Explainer Model: Qwen 72B (Qwen2.5-72B-Instruct)"
echo "  • Layer: 4 only (for testing)"
echo "  • Financial dataset (Yahoo Finance) for domain consistency"
echo "  • Chat template support for proper explanations"
echo "  • F1 scores, precision, recall, accuracy, and specificity metrics"
echo "📁 Results will be saved to: single_layer_results_qwen72b_financial/"
echo "🎯 Using financial dataset: jyanimaulik/yahoo_finance_stockmarket_news"
echo "📊 Using Qwen 72B chat-enabled model: Qwen/Qwen2.5-72B-Instruct"
echo "🔧 Parameters: N_TOKENS=50000, CACHE_CTX_LEN=2048"
echo "🐍 Activating conda environment: sae"
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae

# Set environment variables for chat-enabled model with multi-GPU support
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="2,3,4,5"  # Use 4 free GPUs
export VLLM_USE_DEEP_GEMM=1
export VLLM_GPU_MEMORY_UTILIZATION=0.25  # Reduced for memory constraints
export VLLM_MAX_MODEL_LEN=4096
export VLLM_BLOCK_SIZE=16
export VLLM_SWAP_SPACE=0

# Parameters
BASE_MODEL="meta-llama/Llama-2-7b-hf"
EXPLAINER_MODEL="Qwen/Qwen2.5-14B-Instruct"
LAYER=4
N_TOKENS=50000
CACHE_CTX_LEN=2048
BATCH_SIZE=2
NUM_GPUS=2
EXPLAINER_MODEL_MAX_LEN=4096
NUM_EXAMPLES_PER_SCORER_PROMPT=8
N_NON_ACTIVATING=25
MIN_EXAMPLES=30
DATASET_SPLIT="train[:15%]"
RESULTS_DIR="single_layer_results_qwen72b_financial"
RUN_NAME="single_layer_qwen72b_financial_layer${LAYER}"

echo "🔍 Analyzing Layer ${LAYER} with AutoInterp (Qwen 72B + Financial Dataset)..."
echo "----------------------------------------"

# Get top 10 features for the layer
echo "📊 Getting top 10 features for layer ${LAYER}..."
if [ -f "multi_layer_lite_results/features_layer${LAYER}.csv" ]; then
    # Extract top 10 features from the CSV file
    FEATURES=$(python3 -c "
import pandas as pd
df = pd.read_csv('multi_layer_lite_results/features_layer${LAYER}.csv')
top_features = df.head(10)['feature'].tolist()
print(' '.join(map(str, top_features)))
")
    echo "🎯 Top 10 features for layer ${LAYER}: $FEATURES"
else
    echo "❌ Features file not found: multi_layer_lite_results/features_layer${LAYER}.csv"
    echo "Using default features: 141 127 384 3 90 1 156 25 373 2"
    FEATURES="141 127 384 3 90 1 156 25 373 2"
fi

# Change to autointerp directory (like the working script)
cd ../../autointerp/autointerp_full

# Run AutoInterp analysis
echo "🚀 Starting AutoInterp analysis for layer ${LAYER}..."
python -m autointerp_full \
    $BASE_MODEL \
    "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun" \
    --n_tokens $N_TOKENS \
    --cache_ctx_len $CACHE_CTX_LEN \
    --batch_size $BATCH_SIZE \
    --hookpoints layers.${LAYER} \
    --scorers detection \
    --explainer_model $EXPLAINER_MODEL \
    --explainer_provider offline \
    --explainer_model_max_len $EXPLAINER_MODEL_MAX_LEN \
    --num_gpus $NUM_GPUS \
    --num_examples_per_scorer_prompt $NUM_EXAMPLES_PER_SCORER_PROMPT \
    --n_non_activating $N_NON_ACTIVATING \
    --min_examples $MIN_EXAMPLES \
    --non_activating_source FAISS \
    --faiss_embedding_model sentence-transformers/all-MiniLM-L6-v2 \
    --faiss_embedding_cache_dir .embedding_cache \
    --faiss_embedding_cache_enabled \
    --dataset_repo jyanimaulik/yahoo_finance_stockmarket_news \
    --dataset_name default \
    --dataset_split $DATASET_SPLIT \
    --filter_bos \
    --verbose \
    --feature_num $FEATURES \
    --name $RUN_NAME

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
                cp "../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR/$RUN_NAME/results_summary.csv" "../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR/results_summary_layer${LAYER}_qwen72b.csv"
                echo "📈 Summary saved: FinanceLabeling/$RESULTS_DIR/results_summary_layer${LAYER}_qwen72b.csv"
            fi
        fi
        
        # Verify that we actually have feature explanations
        FEATURE_COUNT=$(find "../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR/$RUN_NAME/explanations" -name "feature_*.json" 2>/dev/null | wc -l)
        if [ $FEATURE_COUNT -gt 0 ]; then
            echo "✅ Layer $LAYER AutoInterp analysis completed successfully"
            echo "🎯 Successfully analyzed $FEATURE_COUNT features"
        else
            echo "❌ Layer $LAYER AutoInterp analysis failed - no feature explanations generated"
            ANALYSIS_EXIT_CODE=1
        fi
    else
        echo "❌ Layer $LAYER AutoInterp analysis failed - no results directory found"
        ANALYSIS_EXIT_CODE=1
    fi
else
    echo "❌ Layer $LAYER AutoInterp analysis failed with exit code $ANALYSIS_EXIT_CODE"
fi

echo ""
cd ../../InterpUseCases_autointerp/FinanceLabeling

# Clean up any remaining temporary files
echo "🧹 Cleaning up temporary files..."
cd ../../autointerp/autointerp_full
if [ -d "results/$RUN_NAME" ]; then
    rm -rf "results/$RUN_NAME"
    echo "🗑️  Cleaned up temporary results"
fi
cd ../../InterpUseCases_autointerp/FinanceLabeling

echo "🏁 Single Layer Analysis Complete!"
echo "=================================="
