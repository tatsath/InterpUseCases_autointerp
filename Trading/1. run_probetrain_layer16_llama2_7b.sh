#!/bin/bash

# ProbeTrain Script for Layer 16 of Llama-2-7b-hf with Financial Dataset
# Runs probetrain on the 16th layer with multi-class classification using financial_three_class.csv

echo "=========================================="
echo "ProbeTrain Financial Layer 16 - Llama-2-7b-hf"
echo "=========================================="

# Set up paths
PROBETRAIN_DIR="/home/nvidia/Documents/Hariom/probetrain"
PROBES_DIR="/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Trading"
DATASET_PATH="$PROBES_DIR/financial_three_class.csv"
OUTPUT_DIR="$PROBES_DIR/probetrain_financial_layer16_results"

# Change to probetrain directory
cd "$PROBETRAIN_DIR"

# Check if probetrain is installed
if ! command -v probetrain &> /dev/null; then
    echo "❌ Error: probetrain command not found"
    echo "Installing probetrain..."
    cd probetrain
    pip install -e .
    cd ..
fi

echo "✅ ProbeTrain found"

# Check if dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "❌ Error: financial_three_class.csv not found at $DATASET_PATH"
    echo "Current directory: $(pwd)"
    echo "Files in Probes directory:"
    ls -la "$PROBES_DIR/"
    exit 1
fi

echo "✅ Found financial dataset: $DATASET_PATH"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "1. Running probetrain on layer 16 of Llama-2-7b-hf with financial dataset..."
echo "Model: meta-llama/Llama-2-7b-hf"
echo "Dataset: financial_three_class.csv"
echo "Layer: 16"
echo "Probe type: multi_class"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run probetrain with specific layer 16 and financial dataset
probetrain \
    --model meta-llama/Llama-2-7b-hf \
    --dataset "$DATASET_PATH" \
    --probe-type multi_class \
    --layers "16" \
    --epochs 100 \
    --lr 0.01 \
    --batch-size 8 \
    --max-samples 1000 \
    --device cuda \
    --output-dir "$OUTPUT_DIR"

# Check if training was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
    
    # Check results
    if [ -d "$OUTPUT_DIR" ]; then
        echo "✅ Results directory created: $OUTPUT_DIR"
        echo "Files in results directory:"
        ls -la "$OUTPUT_DIR/"
        
        if [ -f "$OUTPUT_DIR/results.json" ]; then
            echo "✅ Results JSON file found"
            echo "Results summary:"
            head -20 "$OUTPUT_DIR/results.json"
        else
            echo "❌ Results JSON file not found"
        fi
        
        if [ -f "$OUTPUT_DIR/probe_info.json" ]; then
            echo "✅ Probe info JSON file found"
        else
            echo "❌ Probe info JSON file not found"
        fi
        
        if [ -f "$OUTPUT_DIR/probe_layer_0.pt" ]; then
            echo "✅ Layer 16 probe weights saved (as probe_layer_0.pt)"
        else
            echo "❌ Probe weights not found"
        fi
    else
        echo "❌ Results directory not created"
    fi
else
    echo "❌ Training failed with exit code $?"
    exit 1
fi

echo ""
echo "2. Testing investigation mode on layer 16..."
echo "Command: probetrain --investigate --sentence 'Apple stock surged 5% after strong earnings' --output-dir $OUTPUT_DIR --probe-type multi_class --investigate-layers 16"
echo ""

probetrain \
    --investigate \
    --sentence "Apple stock surged 5% after strong earnings" \
    --output-dir "$OUTPUT_DIR" \
    --probe-type multi_class \
    --investigate-layers "16"

echo ""
echo "=========================================="
echo "ProbeTrain Financial Layer 16 Test completed!"
echo "Results saved in: $OUTPUT_DIR"
echo "=========================================="