#!/bin/bash

# All Layers All Features Analysis with Large Model + Yahoo Finance Dataset
# This script runs AutoInterp Full analysis on all 400 features from layers 4, 10, 16, 22, 28
# using Yahoo Finance dataset and a larger, more capable model for better label consistency
# Usage: ./run_all_layers_all_features_large_model.sh

echo "🚀 All Layers All Features Analysis - Large Model + Yahoo Finance"
echo "================================================================="
echo "🔍 Running detailed analysis on all features from multiple layers:"
echo "  • Layers: 4, 10, 16, 22, 28"
echo "  • All 400 features per layer"
echo "  • Yahoo Finance dataset for financial domain"
echo "  • Large model for better consistency"
echo "  • Improved parameters for higher F1 scores"
echo "  • LLM-based explanations with confidence scores"
echo ""

# Configuration
BASE_MODEL="meta-llama/Llama-2-7b-hf"
SAE_MODEL="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
EXPLAINER_MODEL="Qwen/Qwen2.5-72B-Instruct"  # Using the larger 72B model
EXPLAINER_PROVIDER="offline"
N_TOKENS=20000  # Increased for more data
LAYERS="4 10 16 22 28"

# All 400 features (0-indexed for SAE)
FEATURES="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399"

# Create results directory
RESULTS_DIR="all_layers_financial_results"
mkdir -p "$RESULTS_DIR"

echo "📁 Results will be saved to: $RESULTS_DIR/"
echo "🎯 Features to analyze: All 400 features"
echo "🎯 Layers to analyze: $LAYERS"
echo "🤖 Using large model: $EXPLAINER_MODEL"
echo "📊 Using dataset: Yahoo Finance"
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

# Run AutoInterp Full for all layers with large model
echo "🔍 Analyzing All Layers with AutoInterp Full (Large Model)..."
echo "----------------------------------------"

cd ../../autointerp/autointerp_full

# Create run name
RUN_NAME="all_layers_all_features_large_model_yahoo_finance"

python -m autointerp_full \
    "$BASE_MODEL" \
    "$SAE_MODEL" \
    --n_tokens "$N_TOKENS" \
    --cache_ctx_len 1024 \
    --batch_size 4 \
    --feature_num $FEATURES \
    --hookpoints "layers.4" "layers.10" "layers.16" "layers.22" "layers.28" \
    --scorers detection \
    --explainer_model "$EXPLAINER_MODEL" \
    --explainer_provider "$EXPLAINER_PROVIDER" \
    --explainer_model_max_len 8192 \
    --num_gpus 4 \
    --num_examples_per_scorer_prompt 2 \
    --n_non_activating 15 \
    --min_examples 15 \
    --non_activating_source "FAISS" \
    --faiss_embedding_model "sentence-transformers/all-MiniLM-L6-v2" \
    --faiss_embedding_cache_dir ".embedding_cache" \
    --faiss_embedding_cache_enabled \
    --dataset_repo jyanimaulik/yahoo_finance_stockmarket_news \
    --dataset_name default \
    --dataset_split "train[:20%]" \
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
            echo "📊 Generating CSV summary for all layers..."
            python generate_enhanced_results_csv.py "../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR/$RUN_NAME"
            if [ -f "../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR/$RUN_NAME/results_summary.csv" ]; then
                cp "../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR/$RUN_NAME/results_summary.csv" "../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR/results_summary_all_layers_large_model_yahoo_finance.csv"
                echo "📈 Summary saved: FinanceLabeling/$RESULTS_DIR/results_summary_all_layers_large_model_yahoo_finance.csv"
            fi
        fi
        
        # Verify that we actually have feature explanations
        FEATURE_COUNT=$(find "../../InterpUseCases_autointerp/FinanceLabeling/$RESULTS_DIR/$RUN_NAME/explanations" -name "layers.*_latent*.txt" 2>/dev/null | wc -l)
        if [ $FEATURE_COUNT -gt 0 ]; then
            echo "✅ All Layers AutoInterp Full analysis completed successfully"
            echo "🎯 Successfully analyzed $FEATURE_COUNT features across all layers"
        else
            echo "❌ All Layers AutoInterp Full analysis failed - no feature explanations generated"
            ANALYSIS_EXIT_CODE=1
        fi
    else
        echo "❌ All Layers AutoInterp Full analysis failed - no results directory found"
        ANALYSIS_EXIT_CODE=1
    fi
else
    echo "❌ All Layers AutoInterp Full analysis failed with exit code $ANALYSIS_EXIT_CODE"
fi

echo ""
cd ../../InterpUseCases_autointerp/FinanceLabeling

# Clean up any remaining temporary files
echo "🧹 Cleaning up temporary files..."
cd ../../autointerp/autointerp_full
if [ -d "results/all_layers_all_features_large_model_yahoo_finance" ]; then
    rm -rf "results/all_layers_all_features_large_model_yahoo_finance"
    echo "🗑️  Cleaned up temporary results"
fi
cd ../../InterpUseCases_autointerp/FinanceLabeling

echo "🎯 All Layers All Features Analysis Summary (Large Model)"
echo "========================================================="
echo "📊 Results saved in: $RESULTS_DIR/"

# Check final status based on actual results
if [ -d "$RESULTS_DIR/all_layers_all_features_large_model_yahoo_finance" ] && [ -d "$RESULTS_DIR/all_layers_all_features_large_model_yahoo_finance/explanations" ]; then
    FEATURE_COUNT=$(find "$RESULTS_DIR/all_layers_all_features_large_model_yahoo_finance/explanations" -name "layers.*_latent*.txt" 2>/dev/null | wc -l)
    if [ $FEATURE_COUNT -gt 0 ]; then
        echo "   ✅ all_layers_all_features_large_model_yahoo_finance/ (SUCCESS - $FEATURE_COUNT features analyzed)"
        if [ -f "$RESULTS_DIR/results_summary_all_layers_large_model_yahoo_finance.csv" ]; then
            echo "      📈 results_summary_all_layers_large_model_yahoo_finance.csv"
        fi
        
        echo ""
        echo "📋 Analysis Complete!"
        echo "🔍 Check the following for detailed results:"
        echo "   • explanations/: Human-readable feature explanations for all layers"
        echo "   • scores/detection/: F1 scores and metrics for all layers"
        echo "   • results_summary_all_layers_large_model_yahoo_finance.csv: CSV summary"
        echo ""
        
        # Clean up unnecessary directories to save space
        echo "🧹 Cleaning up unnecessary directories to save space..."
        python3 -c "
import os
import shutil

# Clean up latents and log directories to save space
latents_dir = 'all_layers_financial_results/all_layers_all_features_large_model_yahoo_finance/latents'
log_dir = 'all_layers_financial_results/all_layers_all_features_large_model_yahoo_finance/log'
if os.path.exists(latents_dir):
    shutil.rmtree(latents_dir)
    print('🗑️  Removed latents directory')
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
    print('🗑️  Removed log directory')
print('✅ Cleanup completed')
"
        
        echo ""
        echo "✅ All Layers All Features analysis with large model completed successfully!"
        echo "🎯 Using Qwen 72B model for better label consistency across all layers!"
    else
        echo "   ❌ all_layers_all_features_large_model_yahoo_finance/ (FAILED - no feature explanations found)"
        echo ""
        echo "❌ Analysis failed - no feature explanations were generated"
        echo "🔍 Check the logs above for error details"
        exit 1
    fi
else
    echo "   ❌ all_layers_all_features_large_model_yahoo_finance/ (FAILED - no results directory found)"
    echo ""
    echo "❌ Analysis failed - no results were generated"
    echo "🔍 Check the logs above for error details"
    exit 1
fi
