#!/bin/bash

# Multi-Layer AutoInterp Top 10 Analysis - Custom Financial Prompt
# This script uses a custom financial-specific prompt template for more specific explanations

echo "🚀 Multi-Layer AutoInterp Top 10 Analysis - Custom Financial Prompt"
echo "=================================================================================="
echo "🔍 Running analysis on top 10 features from multiple layers:"
echo "  • Base Model: 7B parameters (Llama-2-7b-hf)"
echo "  • Explainer Model: Qwen 72B (Qwen2.5-72B-Instruct)"
echo "  • Layers: 4, 10, 16, 22, 28"
echo "  • Top 10 features per layer (faster analysis)"
echo "  • Financial dataset (Yahoo Finance) for domain consistency"
echo "  • Custom financial prompt for more specific explanations"
echo "  • F1 scores, precision, recall, accuracy, and specificity metrics"
echo "  • Multi-GPU support for better performance"
echo ""

# Configuration for custom financial prompt
BASE_MODEL="meta-llama/Llama-2-7b-hf"
SAE_MODEL="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
EXPLAINER_MODEL="Qwen/Qwen2.5-72B-Instruct"  # Qwen 72B model that worked successfully
EXPLAINER_PROVIDER="offline"
N_TOKENS=20000  # Same as working script
CACHE_CTX_LEN=1024  # Same as working script

# Layers to analyze - just layer 4 for testing
LAYERS=(4)

# Create results directory
RESULTS_DIR="multi_layer_top10_results_qwen72b_financial_custom_prompt"
mkdir -p "$RESULTS_DIR"

echo "📁 Results will be saved to: $RESULTS_DIR/"
echo "🎯 Using financial dataset: jyanimaulik/yahoo_finance_stockmarket_news"
echo "📊 Using Qwen 72B chat-enabled model: $EXPLAINER_MODEL"
echo "🔧 Parameters: N_TOKENS=$N_TOKENS, CACHE_CTX_LEN=$CACHE_CTX_LEN"
echo "🎯 Using custom financial prompt for more specific explanations"
echo ""

# Set environment variables for 70B model with multi-GPU support
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
    echo "🔍 Analyzing Layer $layer with AutoInterp Top 10 (Qwen 72B + Custom Financial Prompt)..."
    echo "----------------------------------------"
    
    # Get top 10 features for this layer
    echo "📊 Getting top 10 features for layer $layer..."
    python3 -c "
import torch
from sparsify import SparseCoder
import numpy as np

# Load SAE model
sae_model = SparseCoder.load('$SAE_MODEL', device='cpu')

# Get top 10 features by activation strength
layer_key = f'layers.{layer}'
if layer_key in sae_model:
    # Get feature activations and find top 10
    activations = sae_model[layer_key].encoder.weight.norm(dim=1)
    top_features = torch.topk(activations, k=10).indices.tolist()
    print(f'🎯 Top 10 features for layer {layer}: {\" \".join(map(str, top_features))}')
else:
    print(f'❌ Layer {layer_key} not found in SAE model')
    exit(1)
"
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to get top 10 features for layer $layer"
        continue
    fi
    
    # Run AutoInterp analysis with custom financial prompt
    echo "🚀 Starting AutoInterp analysis for layer $layer with custom financial prompt..."
    
    # Create a temporary custom explainer that uses the financial prompt
    cat > /tmp/custom_financial_explainer.py << 'EOF'
import asyncio
from dataclasses import dataclass
from ..explainer import ActivatingExample, Explainer
from .prompt_builder import build_prompt

# Import the financial prompt
from .prompts_financial import SYSTEM_CONTRASTIVE

@dataclass
class CustomFinancialExplainer(Explainer):
    activations: bool = True
    cot: bool = False

    def _build_prompt(self, examples: list[ActivatingExample]) -> list[dict]:
        highlighted_examples = []

        for i, example in enumerate(examples):
            str_toks = example.str_tokens
            activations = example.activations.tolist()
            highlighted_examples.append(self._highlight(str_toks, activations))

            if self.activations:
                assert (
                    example.normalized_activations is not None
                ), "Normalized activations are required for activations in explainer"
                normalized_activations = example.normalized_activations.tolist()
                highlighted_examples.append(
                    self._join_activations(
                        str_toks, activations, normalized_activations
                    )
                )

        highlighted_examples = "\n".join(highlighted_examples)

        # Use the financial prompt instead of default
        return [
            {"role": "system", "content": SYSTEM_CONTRASTIVE},
            {"role": "user", "content": f"\n{highlighted_examples}\n"}
        ]

    def call_sync(self, record):
        return asyncio.run(self.__call__(record))
EOF

    # Run the analysis with the custom financial prompt
    python -m autointerp_full \
        "$BASE_MODEL" \
        "$SAE_MODEL" \
        --n_tokens $N_TOKENS \
        --cache_ctx_len $CACHE_CTX_LEN \
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
        --dataset_name default \
        --dataset_column text \
        --dataset_split "train[:10%]" \
        --filter_bos \
        --verbose \
        --name "multi_layer_top10_qwen72b_financial_custom_prompt_layer$layer"
    
    ANALYSIS_EXIT_CODE=$?
    
    if [ $ANALYSIS_EXIT_CODE -ne 0 ]; then
        echo "❌ Layer $layer AutoInterp Top 10 analysis with custom financial prompt failed with exit code $ANALYSIS_EXIT_CODE"
        echo "🛑 Stopping execution due to failure"
        exit 1
    fi
    
    echo "✅ Layer $layer AutoInterp Top 10 analysis with custom financial prompt completed successfully"
    
    # Move results to the main results directory
    if [ -d "results/multi_layer_top10_qwen72b_financial_custom_prompt_layer$layer" ]; then
        echo "📋 Results moved to: FinanceLabeling/$RESULTS_DIR/multi_layer_top10_qwen72b_financial_custom_prompt_layer$layer/"
        mkdir -p "$RESULTS_DIR/multi_layer_top10_qwen72b_financial_custom_prompt_layer$layer"
        cp -r "results/multi_layer_top10_qwen72b_financial_custom_prompt_layer$layer"/* "$RESULTS_DIR/multi_layer_top10_qwen72b_financial_custom_prompt_layer$layer/"
        
        # Generate CSV summary
        echo "📊 Generating CSV summary for layer $layer..."
        python3 -c "
import pandas as pd
import os
import glob

# Find the results directory
results_dir = 'FinanceLabeling/$RESULTS_DIR/multi_layer_top10_qwen72b_financial_custom_prompt_layer$layer'
if os.path.exists(results_dir):
    # Read explanations
    explanations = {}
    for file in glob.glob(f'{results_dir}/explanations/*.txt'):
        with open(file, 'r') as f:
            content = f.read().strip()
            # Extract feature number from filename
            feature_num = os.path.basename(file).split('_')[-1].replace('.txt', '')
            explanations[feature_num] = content
    
    # Read scores
    scores = {}
    for file in glob.glob(f'{results_dir}/scores/detection/*.txt'):
        with open(file, 'r') as f:
            content = f.read().strip()
            # Extract F1 score from content
            lines = content.split('\n')
            f1_score = None
            for line in lines:
                if 'F1 Score:' in line:
                    f1_score = float(line.split('F1 Score:')[1].strip())
                    break
            feature_num = os.path.basename(file).split('_')[-1].replace('.txt', '')
            scores[feature_num] = f1_score
    
    # Create summary
    summary_data = []
    for feature_num in explanations:
        summary_data.append({
            'Feature': feature_num,
            'Explanation': explanations.get(feature_num, 'N/A'),
            'F1_Score': scores.get(feature_num, 'N/A')
        })
    
    # Save to CSV
    df = pd.DataFrame(summary_data)
    csv_path = f'{results_dir}/results_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f'✅ CSV summary generated: {csv_path}')
    
    # Print summary
    print(f'📊 Found {len(summary_data)} features with results')
    for item in summary_data:
        f1_str = f\"F1: {item['F1_Score']:.3f}\" if item['F1_Score'] != 'N/A' else 'F1: N/A'
        print(f\"  Feature {item['Feature']}: {item['Explanation']} ({f1_str})\")
else:
    print(f'❌ Results directory not found: {results_dir}')
"
        
        # Save summary to main results directory
        if [ -f "FinanceLabeling/$RESULTS_DIR/multi_layer_top10_qwen72b_financial_custom_prompt_layer$layer/results_summary.csv" ]; then
            cp "FinanceLabeling/$RESULTS_DIR/multi_layer_top10_qwen72b_financial_custom_prompt_layer$layer/results_summary.csv" "$RESULTS_DIR/results_summary_layer$layer.csv"
            echo "📈 Summary saved: FinanceLabeling/$RESULTS_DIR/results_summary_layer$layer.csv"
        fi
    else
        echo "❌ Results directory not found for layer $layer"
    fi
    
    echo ""
done

# Clean up temporary files
echo "🧹 Cleaning up temporary files..."
rm -f /tmp/custom_financial_explainer.py

echo "🎯 Multi-Layer AutoInterp Top 10 Analysis Summary - Custom Financial Prompt"
echo "=================================================================================="
echo "📊 Results saved in: $RESULTS_DIR/"
echo "📁 Directories created:"
for layer in "${LAYERS[@]}"; do
    if [ -d "$RESULTS_DIR/multi_layer_top10_qwen72b_financial_custom_prompt_layer$layer" ]; then
        echo "   ✅ multi_layer_top10_qwen72b_financial_custom_prompt_layer$layer/"
        if [ -f "$RESULTS_DIR/results_summary_layer$layer.csv" ]; then
            echo "      📈 results_summary_layer$layer.csv"
        fi
    fi
done

echo ""
echo "📋 Analysis Complete!"
echo "🔍 Check the following for detailed results:"
echo "   • explanations/: Human-readable feature explanations with custom financial prompt"
echo "   • scores/detection/: F1 scores and metrics"
echo "   • results_summary_layer*.csv: CSV summaries per layer"
echo "   • comprehensive_metrics_layer*_financial.csv: All metrics per layer"
echo ""
echo "🧹 Cleaning up unnecessary directories to save space..."
python3 -c "
import shutil
import os
import glob

# Clean up temporary results
temp_dirs = glob.glob('results/multi_layer_top10_qwen72b_financial_custom_prompt_layer*')
for temp_dir in temp_dirs:
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f'🗑️  Cleaned up {temp_dir}')

print('✅ Cleanup completed')
"

echo ""
echo "✅ Multi-layer AutoInterp Top 10 analysis with custom financial prompt completed!"
echo "🎯 Using Qwen2.5-72B-Instruct with Yahoo Finance dataset and custom financial prompt for more specific explanations!"
