#!/bin/bash

# Multi-Layer AutoInterp Full Analysis - All 400 Features
# This script runs AutoInterp Full analysis on ALL 400 features from multiple layers
# Usage: ./run_multi_layer_full_analysis.sh

echo "🚀 Multi-Layer AutoInterp Full Analysis - All 400 Features"
echo "============================================================"
echo "🔍 Running detailed analysis on ALL 400 features from multiple layers:"
echo "  • Layers: 4, 10, 16, 22, 28"
echo "  • ALL 400 features per layer (0-399)"
echo "  • Improved parameters for better F1 scores"
echo "  • LLM-based explanations with confidence scores"
echo "  • F1 scores, precision, and recall metrics"
echo ""

# Configuration
BASE_MODEL="meta-llama/Llama-2-7b-hf"
SAE_MODEL="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
EXPLAINER_MODEL="Qwen/Qwen2.5-7B-Instruct"
EXPLAINER_PROVIDER="offline"
N_TOKENS=15000  # Increased from 10000

# Layers to analyze
LAYERS=(4 10 16 22 28)

# Create results directory
RESULTS_DIR="multi_layer_full_results"
mkdir -p "$RESULTS_DIR"

echo "📁 Results will be saved to: $RESULTS_DIR/"
echo ""

# Activate conda environment for SAE
echo "🐍 Activating conda environment: sae"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae
echo ""

# Set environment variables for better performance (improved parameters)
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="1,2,3,4"  # Use 4 GPUs for better performance
export VLLM_USE_DEEP_GEMM=1
export VLLM_GPU_MEMORY_UTILIZATION=0.7
export VLLM_MAX_MODEL_LEN=4096
export VLLM_BLOCK_SIZE=16
export VLLM_SWAP_SPACE=0

# Run analysis for each layer
for layer in "${LAYERS[@]}"; do
    echo "🔍 Analyzing Layer $layer with AutoInterp Full (ALL 400 features)..."
    echo "----------------------------------------"
    
    # Generate all 400 features (0-399)
    echo "📋 Generating all 400 features for layer $layer"
    FEATURES=$(seq 0 399 | tr '\n' ' ')
    
    echo "🎯 Features to analyze: ALL 400 features (0-399)"
    
    # Run AutoInterp Full for this layer
    cd ../../autointerp/autointerp_full
    
    # Create layer-specific run name
    RUN_NAME="multi_layer_full_layer${layer}"
    
    python -m autointerp_full \
        "$BASE_MODEL" \
        "$SAE_MODEL" \
        --n_tokens "$N_TOKENS" \
        --cache_ctx_len 512 \
        --batch_size 8 \
        --feature_num $FEATURES \
        --hookpoints "layers.$layer" \
        --scorers detection \
        --explainer_model "$EXPLAINER_MODEL" \
        --explainer_provider "$EXPLAINER_PROVIDER" \
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
    if [ $? -eq 0 ]; then
        echo "✅ Layer $layer AutoInterp Full analysis completed successfully"
        
        # Move results to our organized directory (like OpenRouter script)
        if [ -d "results/$RUN_NAME" ]; then
            # Remove existing results directory if it exists
            rm -rf "../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR/$RUN_NAME"
            # Move the entire results directory
            mv "results/$RUN_NAME" "../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR/"
            echo "📋 Results moved to: FinanceLabeling/$RESULTS_DIR/$RUN_NAME/"
            
            # Generate CSV summary if results exist
            if [ -d "../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR/$RUN_NAME/explanations" ] && [ "$(ls -A ../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR/$RUN_NAME/explanations)" ]; then
                echo "📊 Generating CSV summary for layer $layer..."
                python generate_results_csv.py "../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR/$RUN_NAME"
                if [ -f "../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR/$RUN_NAME/results_summary.csv" ]; then
                    cp "../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR/$RUN_NAME/results_summary.csv" "../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR/results_summary_layer${layer}.csv"
                    echo "📈 Summary saved: FinanceLabeling/$RESULTS_DIR/results_summary_layer${layer}.csv"
                fi
            fi
        fi
    else
        echo "❌ Layer $layer AutoInterp Full analysis failed"
    fi
    
    echo ""
    cd ../../InterpUseCases_autointerp/FinanceLabeling
done

# Clean up any remaining temporary files (like OpenRouter script)
echo "🧹 Cleaning up temporary files..."
cd ../../autointerp/autointerp_full
for layer in "${LAYERS[@]}"; do
    if [ -d "results/multi_layer_full_layer${layer}" ]; then
        rm -rf "results/multi_layer_full_layer${layer}"
        echo "🗑️  Cleaned up temporary results for layer $layer"
    fi
done
cd ../../InterpUseCases_autointerp/FinanceLabeling

echo "🎯 Multi-Layer AutoInterp Full Analysis Summary (ALL 400 Features)"
echo "=================================================================="
echo "📊 Results saved in: $RESULTS_DIR/"
echo "📁 Directories created:"
for layer in "${LAYERS[@]}"; do
    if [ -d "$RESULTS_DIR/multi_layer_full_layer${layer}" ]; then
        echo "   ✅ multi_layer_full_layer${layer}/ (ALL 400 features)"
        if [ -f "$RESULTS_DIR/results_summary_layer${layer}.csv" ]; then
            echo "      📈 results_summary_layer${layer}.csv"
            # Count features in CSV
            FEATURE_COUNT=$(wc -l < "$RESULTS_DIR/results_summary_layer${layer}.csv" 2>/dev/null || echo "0")
            echo "      📊 Features analyzed: $((FEATURE_COUNT - 1))"  # Subtract header
        fi
    else
        echo "   ❌ multi_layer_full_layer${layer}/ (failed)"
    fi
done

echo ""
echo "📋 Analysis Complete!"
echo "🔍 Check the following for detailed results:"
echo "   • explanations/: Human-readable feature explanations"
echo "   • scores/detection/: F1 scores and metrics"
echo "   • results_summary_layer*.csv: CSV summaries per layer"
echo ""
echo "🧹 Cleaning up unnecessary directories to save space..."
python3 -c "
import os
import shutil

# Clean up latents and log directories to save space for each layer
layers = [4, 10, 16, 22, 28]
for layer in layers:
    latents_dir = f'multi_layer_full_results/multi_layer_full_layer{layer}/latents'
    log_dir = f'multi_layer_full_results/multi_layer_full_layer{layer}/log'
    if os.path.exists(latents_dir):
        shutil.rmtree(latents_dir)
        print(f'🗑️  Removed latents directory for layer {layer}')
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        print(f'🗑️  Removed log directory for layer {layer}')
print('✅ Cleanup completed')
"

echo ""
echo "✅ Multi-layer AutoInterp Full analysis completed!"
echo "🎯 All 400 features analyzed for each layer with improved parameters!"
echo "📊 CSV files saved: results_summary_layer*.csv"
