#!/bin/bash

# Speed-Optimized Sparse AutoInterp Analysis - Layer 4
# Uses Accelerate with optimized multi-GPU loading
# Much faster loading with init_empty_weights + load_checkpoint_and_dispatch

set -e

echo "⚡ Speed-Optimized Sparse AutoInterp Analysis - Layer 4"
echo "======================================================"
echo "🔍 Running FAST TEST analysis on 10 features of layer 4:"
echo "  • Layer: 4"
echo "  • Features: 0-9 (first 10 features)"
echo "  • Domain: General (sparse features)"
echo "  • Fast Transformers with 4-bit quantization + Flash Attention"
echo "  • Llama-2-7B-Chat with 4-bit quantization + Flash Attention"
echo "  • Optimized for maximum performance"

# Configuration - Using Fast Transformers with 4-bit quantization
BASE_MODEL="meta-llama/Llama-2-7b-hf"
SAE_MODEL="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
EXPLAINER_MODEL="meta-llama/Llama-2-7b-chat-hf"  # Fast 7B model with 4-bit quantization
EXPLAINER_PROVIDER="transformers_fast"  # Use Fast Transformers with optimizations
N_TOKENS=5000  # Increased for better analysis
LAYER=4
FEATURES="0 1 2 3 4 5 6 7 8 9"  # Test with first 10 features
RESULTS_DIR="sparse_layer4_speed_optimized_results"
RUN_NAME="sparse_layer4_speed_optimized"

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
echo "🔍 Analyzing Layer 4 with Speed-Optimized AutoInterp..."
echo "----------------------------------------"
echo "📊 Configuration Details:"
echo "  • Base Model: $BASE_MODEL"
echo "  • SAE Model: $SAE_MODEL"
echo "  • Explainer Model: $EXPLAINER_MODEL"
echo "  • Provider: $EXPLAINER_PROVIDER"
echo "  • Max Model Length: 2048"
echo "  • Number of GPUs: 8"
echo "  • Features: $FEATURES"
echo ""
echo "🚀 Starting analysis with verbose logging..."
echo "----------------------------------------"

# Set environment variables for Transformers multi-GPU optimization
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  # Use all 8 GPUs
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

# H100 optimization settings for Transformers
export TORCH_BACKENDS_CUDA_MATMUL_ALLOW_TF32=1
export TORCH_FLOAT32_MATMUL_PRECISION=high

# Verbose logging settings
export TRANSFORMERS_VERBOSITY=info
export TOKENIZERS_VERBOSITY=info
export ACCELERATE_LOG_LEVEL=info

# Speed-optimized analysis
echo "📁 Changing to autointerp_full directory..."
cd /home/nvidia/Documents/Hariom/autointerp/autointerp_full

echo "🐍 Starting Python analysis with verbose output..."
echo "⏱️  This is a FAST TEST - should complete in 2-3 minutes..."
echo ""
echo "🔍 If it gets stuck at 'Processing items: 0it', it means:"
echo "  • Dataset might be too small (try increasing dataset_split)"
echo "  • Model might be running out of memory"
echo "  • Pipeline might be waiting for data"
echo ""

# Run with progress monitoring
python -m autointerp_full \
    "$BASE_MODEL" \
    "$SAE_MODEL" \
    --n_tokens 5000 \
    --feature_num $FEATURES \
    --hookpoints "layers.$LAYER" \
    --scorers detection \
    --explainer_model "$EXPLAINER_MODEL" \
    --explainer_provider "$EXPLAINER_PROVIDER" \
    --explainer_model_max_len 512 \
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

# Capture exit code
EXIT_CODE=$?

echo ""
echo "🔍 Analysis completed with exit code: $EXIT_CODE"

# Check if analysis completed successfully
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Speed-Optimized Sparse AutoInterp Analysis Completed Successfully!"
    echo "=================================================================="
    echo "📊 Results Summary:"
    echo "  • Layer: $LAYER"
    echo "  • Features analyzed: $FEATURES"
    echo "  • Results directory: $RESULTS_DIR/"
    echo "  • Backend: $EXPLAINER_PROVIDER (Fast Transformers with 4-bit quantization)"
    echo "  • GPUs used: 8 (optimized for speed)"
    echo ""
    echo "⚡ Analysis completed with speed optimizations!"
    echo "   Check the results directory for detailed outputs."
else
    echo ""
    echo "❌ Speed-Optimized Sparse AutoInterp Analysis Failed!"
    echo "==================================================="
    echo "🔍 Troubleshooting:"
    echo "  • Check GPU memory availability"
    echo "  • Verify model paths are correct"
    echo "  • Ensure all dependencies are installed"
    echo "  • Try reducing --num_gpus if memory issues persist"
    echo ""
    echo "💡 For even faster performance, try:"
    echo "  • Use 4-bit quantization for memory efficiency"
    echo "  • Enable flash_attention_2 for H100 speedups"
    echo "  • Use device_map='auto' for automatic sharding"
    exit 1
fi

echo ""
echo "🧹 Cleaning up temporary files..."
# Clean up any temporary files
rm -f .embedding_cache/*.tmp 2>/dev/null || true

echo "✅ Speed-Optimized Analysis Complete!"

