#!/bin/bash

# Multi-Layer AutoInterp Top 10 Analysis - Chat-Enabled Model + Financial Dataset
# This script runs AutoInterp analysis on top 10 features from multiple layers using a chat-enabled model
# Usage: ./run_multi_layer_top10_analysis_70b_financial.sh

echo "🚀 Multi-Layer AutoInterp Top 10 Analysis - Chat-Enabled Model + Financial Dataset"
echo "=================================================================================="
echo "🔍 Running analysis on top 10 features from multiple layers:"
echo "  • Base Model: 7B parameters (Llama-2-7b-hf)"
echo "  • Explainer Model: Chat-enabled model (Qwen2.5-7B-Instruct)"
echo "  • Layers: 4, 10, 16, 22, 28"
echo "  • Top 10 features per layer (faster analysis)"
echo "  • Financial dataset (Yahoo Finance) for domain consistency"
echo "  • Chat template support for proper explanations"
echo "  • F1 scores, precision, recall, accuracy, and specificity metrics"
echo "  • Multi-GPU support for better performance"
echo ""

# Configuration for chat-enabled model with financial dataset
BASE_MODEL="meta-llama/Llama-2-7b-hf"
SAE_MODEL="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
EXPLAINER_MODEL="Qwen/Qwen2.5-72B-Instruct"  # Qwen 72B model that worked successfully
EXPLAINER_PROVIDER="offline"
N_TOKENS=50000  # Same as working script
CACHE_CTX_LEN=1024  # Same as working script

# Layers to analyze - just layer 4 for testing
LAYERS=(4)

# Create results directory
RESULTS_DIR="multi_layer_top10_results_qwen72b_financial_run2"
mkdir -p "$RESULTS_DIR"

echo "📁 Results will be saved to: $RESULTS_DIR/"
echo "🎯 Using financial dataset: jyanimaulik/yahoo_finance_stockmarket_news"
echo "📊 Using Qwen 72B chat-enabled model: $EXPLAINER_MODEL"
echo "🔧 Parameters: N_TOKENS=$N_TOKENS, CACHE_CTX_LEN=$CACHE_CTX_LEN"
echo ""

# Activate conda environment for SAE
echo "🐍 Activating conda environment: sae"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae
echo ""

# Set environment variables for chat-enabled model with multi-GPU support
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="1,2,3,4"  # Use 4 free GPUs (excluding GPU 0)
export VLLM_USE_DEEP_GEMM=1
export VLLM_GPU_MEMORY_UTILIZATION=0.7
export VLLM_MAX_MODEL_LEN=4096  # Same as working script
export VLLM_BLOCK_SIZE=32
export VLLM_SWAP_SPACE=0

# Run analysis for each layer
for layer in "${LAYERS[@]}"; do
    echo "🔍 Analyzing Layer $layer with AutoInterp Top 10 (Qwen 72B + Financial Dataset)..."
    echo "----------------------------------------"
    
    # Get top 10 features for this layer
    echo "📊 Getting top 10 features for layer $layer..."
    FEATURES=$(python3 -c "
import pandas as pd
try:
    df = pd.read_csv('multi_layer_lite_results/features_layer${layer}.csv')
    if 'feature' in df.columns:
        features = df['feature'].head(10).tolist()
    else:
        features = df.iloc[:10, 1].tolist()
    print(' '.join(map(str, features)))
except Exception as e:
    print('Error reading features:', e)
    # Fallback to first 10 features if file doesn't exist
    print(' '.join(map(str, range(0, 10))))
")
    
    echo "🎯 Top 10 features for layer $layer: $FEATURES"
    
    # Run AutoInterp for this layer with top 10 features
    cd ../../autointerp/autointerp_full
    
    # Create layer-specific run name
    RUN_NAME="multi_layer_top10_qwen72b_financial_layer${layer}"
    
    python -m autointerp_full \
        "$BASE_MODEL" \
        "$SAE_MODEL" \
        --n_tokens "$N_TOKENS" \
        --cache_ctx_len "$CACHE_CTX_LEN" \
        --batch_size 4 \
        --hookpoints "layers.$layer" \
        --scorers detection \
        --explainer_model "$EXPLAINER_MODEL" \
        --explainer_provider "$EXPLAINER_PROVIDER" \
        --explainer_model_max_len 4096 \
        --num_gpus 4 \
        --num_examples_per_scorer_prompt 10 \
        --n_non_activating 30 \
        --min_examples 50 \
        --non_activating_source "FAISS" \
        --faiss_embedding_model "sentence-transformers/all-MiniLM-L6-v2" \
        --faiss_embedding_cache_dir ".embedding_cache" \
        --faiss_embedding_cache_enabled \
        --dataset_repo jyanimaulik/yahoo_finance_stockmarket_news \
        --dataset_name "default" \
        --dataset_split "train[:15%]" \
        --filter_bos \
        --verbose \
        --feature_num $FEATURES \
        --name "$RUN_NAME"
    
    # Check if analysis was successful
    ANALYSIS_EXIT_CODE=$?
    if [ $ANALYSIS_EXIT_CODE -ne 0 ]; then
        echo "❌ Layer $layer AutoInterp Top 10 analysis with Qwen 72B failed with exit code $ANALYSIS_EXIT_CODE"
        echo "🛑 Stopping execution due to failure"
        exit 1
    fi
    
    if [ $ANALYSIS_EXIT_CODE -eq 0 ]; then
        echo "✅ Layer $layer AutoInterp Top 10 analysis with Qwen 72B completed successfully"
        
        # Move results to our organized directory
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
                
                # Generate comprehensive metrics CSV with all metrics
                echo "📊 Generating comprehensive metrics CSV for layer $layer..."
                python3 -c "
import json
import os
import csv

# Generate comprehensive metrics CSV
scores_dir = '../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR/$RUN_NAME/scores/detection/'
if os.path.exists(scores_dir):
    # Read existing labels
    labels = {}
    summary_file = '../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR/$RUN_NAME/results_summary.csv'
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels[int(row['feature'])] = row['label']
    
    # Get all feature files
    feature_files = [f for f in os.listdir(scores_dir) if f.endswith('.txt')]
    results = []
    
    for feature_file in feature_files:
        # Extract feature number from filename
        feature_num = int(feature_file.split('_')[-1].split('.')[0])
        
        with open(os.path.join(scores_dir, feature_file), 'r') as f:
            data = json.load(f)
        
        # Calculate confusion matrix
        tp = sum(1 for item in data if item['correct'] and item['prediction'])
        fp = sum(1 for item in data if not item['correct'] and item['prediction'])
        tn = sum(1 for item in data if item['correct'] and not item['prediction'])
        fn = sum(1 for item in data if not item['correct'] and not item['prediction'])
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        results.append({
            'layer': $layer,
            'feature': feature_num,
            'label': labels.get(feature_num, ''),
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'accuracy': accuracy,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        })
    
    # Write comprehensive metrics CSV
    output_file = '../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR/comprehensive_metrics_layer${layer}_financial.csv'
    with open(output_file, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    print(f'📊 Comprehensive metrics saved: comprehensive_metrics_layer${layer}_financial.csv')
    print(f'   Features analyzed: {len(results)}')
    if results:
        print(f'   Average F1: {sum(r[\"f1_score\"] for r in results)/len(results):.3f}')
        print(f'   Average Accuracy: {sum(r[\"accuracy\"] for r in results)/len(results):.3f}')
        print(f'   Average Specificity: {sum(r[\"specificity\"] for r in results)/len(results):.3f}')
"
            fi
        fi
    else
        echo "❌ Layer $layer AutoInterp Top 10 analysis with Qwen 72B failed with exit code $ANALYSIS_EXIT_CODE"
    fi
    
    echo ""
    cd ../../InterpUseCases_autointerp/FinanceLabeling
done

# Clean up any remaining temporary files
echo "🧹 Cleaning up temporary files..."
cd ../../autointerp/autointerp_full
for layer in "${LAYERS[@]}"; do
    if [ -d "results/multi_layer_top10_qwen72b_financial_layer${layer}" ]; then
        rm -rf "results/multi_layer_top10_qwen72b_financial_layer${layer}"
        echo "🗑️  Cleaned up temporary results for layer $layer"
    fi
done
cd ../../InterpUseCases_autointerp/FinanceLabeling

echo "🎯 Multi-Layer AutoInterp Top 10 Analysis Summary - Qwen 72B + Financial Dataset"
echo "=================================================================================="
echo "📊 Results saved in: $RESULTS_DIR/"
echo "📁 Directories created:"
for layer in "${LAYERS[@]}"; do
    if [ -d "$RESULTS_DIR/multi_layer_top10_qwen72b_financial_layer${layer}" ]; then
        echo "   ✅ multi_layer_top10_qwen72b_financial_layer${layer}/"
        if [ -f "$RESULTS_DIR/results_summary_layer${layer}.csv" ]; then
            echo "      📈 results_summary_layer${layer}.csv"
        fi
        if [ -f "$RESULTS_DIR/comprehensive_metrics_layer${layer}_financial.csv" ]; then
            echo "      📊 comprehensive_metrics_layer${layer}_financial.csv"
        fi
    else
        echo "   ❌ multi_layer_top10_qwen72b_financial_layer${layer}/ (failed)"
    fi
done

echo ""
echo "📋 Analysis Complete!"
echo "🔍 Check the following for detailed results:"
echo "   • explanations/: Human-readable feature explanations"
echo "   • scores/detection/: F1 scores and metrics"
echo "   • results_summary_layer*.csv: CSV summaries per layer"
echo "   • comprehensive_metrics_layer*_financial.csv: All metrics per layer"
echo ""

# Clean up unnecessary directories to save space
echo "🧹 Cleaning up unnecessary directories to save space..."
python3 -c "
import os
import shutil

# Clean up latents and log directories to save space for each layer
layers = [4, 10, 16, 22, 28]
for layer in layers:
    latents_dir = f'multi_layer_top10_results_qwen72b_financial/multi_layer_top10_qwen72b_financial_layer{layer}/latents'
    log_dir = f'multi_layer_top10_results_qwen72b_financial/multi_layer_top10_qwen72b_financial_layer{layer}/log'
    if os.path.exists(latents_dir):
        shutil.rmtree(latents_dir)
        print(f'🗑️  Removed latents directory for layer {layer}')
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        print(f'🗑️  Removed log directory for layer {layer}')
print('✅ Cleanup completed')
"

echo ""
echo "✅ Multi-layer AutoInterp Top 10 analysis with Qwen 72B + financial dataset completed!"
echo "🎯 Using Qwen2.5-72B-Instruct with Yahoo Finance dataset for comprehensive analysis!"
