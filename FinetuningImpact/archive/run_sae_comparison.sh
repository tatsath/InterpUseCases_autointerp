#!/bin/bash

# SAE Model Comparison Analysis
# Compares top 10 features between base and finetuned SAE models
# Usage: ./run_sae_comparison.sh

echo "🚀 SAE Model Comparison Analysis"
echo "================================"
echo "🔍 Comparing SAE models to analyze finetuning impact:"
echo "  • Base Model: llama2_7b_hf_layers4_10_16_22_28_k32_latents400_wikitext103_torchrun"
echo "  • Finetuned Model: llama2_7b_finance_layers4_10_16_22_28_k32_latents400_wikitext103_torchrun"
echo "  • Layers: 4, 10, 16, 22, 28"
echo "  • Top 10 features per layer by activation"
echo "  • Financial domain analysis"
echo ""

# Configuration
BASE_MODEL="meta-llama/Llama-2-7b-hf"
BASE_SAE_MODEL="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4_10_16_22_28_k32_latents400_wikitext103_torchrun"
FINETUNED_SAE_MODEL="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_finance_layers4_10_16_22_28_k32_latents400_wikitext103_torchrun"
DOMAIN_DATA="../../autointerp/autointerp_lite/financial_texts.txt"
GENERAL_DATA="../../autointerp/autointerp_lite/general_texts.txt"
LABELING_MODEL="Qwen/Qwen2.5-7B-Instruct"
TOP_N=10

# Layers to analyze
LAYERS=(4 10 16 22 28)

# Create results directory
RESULTS_DIR="comparison_results"
mkdir -p "$RESULTS_DIR"

echo "📁 Results will be saved to: $RESULTS_DIR/"
echo ""

# Activate conda environment for SAE
echo "🐍 Activating conda environment: sae"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae
echo ""

# Check if models exist
if [ ! -d "$BASE_SAE_MODEL" ]; then
    echo "❌ Base SAE model not found: $BASE_SAE_MODEL"
    exit 1
fi

if [ ! -d "$FINETUNED_SAE_MODEL" ]; then
    echo "❌ Finetuned SAE model not found: $FINETUNED_SAE_MODEL"
    exit 1
fi

echo "✅ Both SAE models found"
echo ""

# Function to run analysis for a model
run_model_analysis() {
    local model_path="$1"
    local model_name="$2"
    local output_suffix="$3"
    
    echo "🔍 Analyzing $model_name..."
    echo "📁 Model path: $model_path"
    echo ""
    
    # Create model-specific results directory
    MODEL_RESULTS_DIR="$RESULTS_DIR/${model_name// /_}"
    mkdir -p "$MODEL_RESULTS_DIR"
    
    # Run analysis for each layer
    for layer in "${LAYERS[@]}"; do
        echo "  📊 Processing Layer $layer..."
        
        # Run AutoInterp Lite for this layer
        cd ../../autointerp/autointerp_lite
        python run_analysis.py \
            --base_model "$BASE_MODEL" \
            --sae_model "$model_path" \
            --domain_data "$DOMAIN_DATA" \
            --general_data "$GENERAL_DATA" \
            --top_n "$TOP_N" \
            --layer_idx "$layer" \
            --enable_labeling \
            --labeling_model "$LABELING_MODEL" \
            --output_dir "../../InterpUseCases_autointerp/FinetuningImpact/$MODEL_RESULTS_DIR"
        
        # Check if analysis was successful
        if [ $? -eq 0 ]; then
            echo "    ✅ Layer $layer analysis completed successfully"
            
            # Find the most recent results directory for this layer
            LATEST_RESULT=$(ls -t results/analysis_*/features_layer${layer}.csv 2>/dev/null | head -1)
            if [ -n "$LATEST_RESULT" ]; then
                # Copy results to our organized directory
                cp "$LATEST_RESULT" "../../InterpUseCases_autointerp/FinetuningImpact/$MODEL_RESULTS_DIR/features_layer${layer}.csv"
                echo "    📋 Results copied to: $MODEL_RESULTS_DIR/features_layer${layer}.csv"
            fi
        else
            echo "    ❌ Layer $layer analysis failed"
        fi
        
        echo ""
        cd ../../InterpUseCases_autointerp/FinetuningImpact
    done
    
    echo "✅ $model_name analysis completed"
    echo ""
}

# Run analysis for both models
run_model_analysis "$BASE_SAE_MODEL" "Base_Model" "base"
run_model_analysis "$FINETUNED_SAE_MODEL" "Finetuned_Model" "finetuned"

# Run comparison analysis
echo "🔄 Running comparison analysis..."
python compare_sae_models.py

# Generate summary report
echo "📋 Generating summary report..."
python -c "
import pandas as pd
import os

# Load results
base_results = {}
finetuned_results = {}

for layer in [4, 10, 16, 22, 28]:
    base_file = f'comparison_results/Base_Model/features_layer{layer}.csv'
    finetuned_file = f'comparison_results/Finetuned_Model/features_layer{layer}.csv'
    
    if os.path.exists(base_file):
        base_results[layer] = pd.read_csv(base_file)
    if os.path.exists(finetuned_file):
        finetuned_results[layer] = pd.read_csv(finetuned_file)

# Generate comparison summary
summary_data = []
for layer in [4, 10, 16, 22, 28]:
    if layer in base_results and layer in finetuned_results:
        base_features = set(base_results[layer]['feature'].tolist())
        finetuned_features = set(finetuned_results[layer]['feature'].tolist())
        
        unique_to_base = base_features - finetuned_features
        unique_to_finetuned = finetuned_features - base_features
        common = base_features & finetuned_features
        
        summary_data.append({
            'layer': layer,
            'base_unique_count': len(unique_to_base),
            'finetuned_unique_count': len(unique_to_finetuned),
            'common_count': len(common),
            'base_unique_features': list(unique_to_base)[:10],
            'finetuned_unique_features': list(unique_to_finetuned)[:10]
        })

# Save summary
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('comparison_summary.csv', index=False)
print('📊 Summary saved to: comparison_summary.csv')
"

echo ""
echo "🎯 SAE Model Comparison Summary"
echo "==============================="
echo "📊 Results saved in: $RESULTS_DIR/"
echo "📁 Files created:"
echo "   📋 comparison_summary.csv - Overall comparison summary"
echo "   📋 sae_model_comparison_report.csv - Detailed feature comparison"
echo "   📁 Base_Model/ - Base model results per layer"
echo "   📁 Finetuned_Model/ - Finetuned model results per layer"
echo ""

# Show summary statistics
if [ -f "comparison_summary.csv" ]; then
    echo "📈 Summary Statistics:"
    python -c "
import pandas as pd
df = pd.read_csv('comparison_summary.csv')
for _, row in df.iterrows():
    print(f'  Layer {row[\"layer\"]}:')
    print(f'    Base model unique: {row[\"base_unique_count\"]} features')
    print(f'    Finetuned model unique: {row[\"finetuned_unique_count\"]} features')
    print(f'    Common features: {row[\"common_count\"]} features')
    print()
"
fi

echo "📋 Next Steps:"
echo "1. Review comparison_summary.csv for overall statistics"
echo "2. Check sae_model_comparison_report.csv for detailed feature analysis"
echo "3. Examine individual layer results in Base_Model/ and Finetuned_Model/ directories"
echo ""
echo "✅ SAE model comparison analysis completed!"
