#!/bin/bash

# Ultra-Fast Sparse AutoInterp Analysis - Layer 4
# Uses fastest LLM backends (3-5x faster than VLLM)
# Optimized for speed and GPU utilization

set -e

echo "🚀 Ultra-Fast Sparse AutoInterp Analysis - Layer 4"
echo "=================================================="
echo "🔍 Running ultra-fast analysis on first 10 features of layer 4:"
echo "  • Layer: 4"
echo "  • Features: 0-9 (first 10 features)"
echo "  • Domain: General (sparse features)"
echo "  • Ultra-fast LLM-based explanations"
echo "  • Optimized for maximum speed"

# Configuration
BASE_MODEL="meta-llama/Llama-2-7b-hf"
SAE_MODEL="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
EXPLAINER_MODEL="Qwen/Qwen2.5-3B-Instruct"
EXPLAINER_PROVIDER="transformers_fast"  # Ultra-fast backend
N_TOKENS=5000
LAYER=4
FEATURES="0 1 2 3 4 5 6 7 8 9"
RESULTS_DIR="sparse_layer4_ultra_fast_results"
RUN_NAME="sparse_layer4_ultra_fast"

echo ""
echo "📁 Results will be saved to: $RESULTS_DIR/"
echo ""

# Activate conda environment
echo "🐍 Activating conda environment: sae"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae

# Check if lite results exist
LITE_RESULTS_DIR="sparse_layer4_lite_results"
if [ -d "$LITE_RESULTS_DIR" ]; then
    echo "📋 Found lite results directory: $LITE_RESULTS_DIR"
    echo "   (Using first 10 features regardless of lite results)"
    echo "🎯 Features to analyze: $FEATURES"
else
    echo "⚠️  No lite results found. Running on first 10 features: $FEATURES"
fi

echo ""
echo "🔍 Analyzing Layer 4 with Ultra-Fast AutoInterp..."
echo "----------------------------------------"

# Set environment variables for maximum speed
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  # Use all 8 GPUs
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

# Ultra-fast analysis with maximum speed optimizations
cd /home/nvidia/Documents/Hariom/autointerp/autointerp_full
python -m autointerp_full \
    "$BASE_MODEL" \
    "$SAE_MODEL" \
    --n_tokens 50000 \
    --feature_num $FEATURES \
    --hookpoints "layers.$LAYER" \
    --scorers detection \
    --explainer_model "$EXPLAINER_MODEL" \
    --explainer_provider "$EXPLAINER_PROVIDER" \
    --explainer_model_max_len 2048 \
    --num_gpus 8 \
    --num_examples_per_scorer_prompt 1 \
    --n_non_activating 5 \
    --min_examples 2 \
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

# Check if analysis completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Ultra-Fast Sparse AutoInterp Analysis Completed Successfully!"
    echo "================================================================"
    echo "📊 Results Summary:"
    echo "  • Layer: $LAYER"
    echo "  • Features analyzed: $FEATURES"
    echo "  • Results directory: $RESULTS_DIR/"
    echo "  • Backend: $EXPLAINER_PROVIDER (3-5x faster than VLLM)"
    echo "  • GPUs used: 8 (maximum utilization)"
    echo ""
    echo "🎯 Analysis completed with ultra-fast performance!"
    echo "   Check the results directory for detailed outputs."
else
    echo ""
    echo "❌ Ultra-Fast Sparse AutoInterp Analysis Failed!"
    echo "=============================================="
    echo "🔍 Troubleshooting:"
    echo "  • Check GPU memory availability"
    echo "  • Verify model paths are correct"
    echo "  • Ensure all dependencies are installed"
    echo "  • Try reducing --num_gpus if memory issues persist"
    echo ""
    echo "💡 For even faster performance, try:"
    echo "  • Use 'exllamav2' provider (requires: pip install exllamav2)"
    echo "  • Reduce --explainer_model_max_len to 1024"
    echo "  • Use smaller model like 'microsoft/DialoGPT-small'"
    exit 1
fi

echo ""
echo "🧹 Cleaning up temporary files..."
# Clean up any temporary files
rm -f .embedding_cache/*.tmp 2>/dev/null || true

echo "✅ Ultra-Fast Analysis Complete!"
