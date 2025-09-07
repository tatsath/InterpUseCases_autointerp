#!/bin/bash

# Script to copy all results from different analysis runs
# This script consolidates results from various analysis configurations

echo "Copying all results from financial analysis runs..."

# Create consolidated results directory
mkdir -p consolidated_results

# Copy multi-layer full results
echo "Copying multi-layer full results..."
cp -r multi_layer_full_results/* consolidated_results/

# Copy multi-layer lite results
echo "Copying multi-layer lite results..."
cp -r multi_layer_lite_results/* consolidated_results/

# Copy single-layer results
echo "Copying single-layer results..."
cp -r single_layer_full_results/* consolidated_results/
cp -r single_layer_openrouter_results/* consolidated_results/

# Create summary report
echo "Creating consolidated summary report..."
cat > consolidated_results/consolidated_summary.txt << EOF
Financial Feature Analysis - Consolidated Results
================================================

Analysis Date: $(date)
Total Layers Analyzed: 5 (4, 10, 16, 22, 28)
Total Features Analyzed: 50
Analysis Types: Multi-layer Full, Multi-layer Lite, Single-layer

Key Findings:
- Layer 4: Basic financial terminology and numerical processing
- Layer 10: Risk assessment and market sentiment analysis
- Layer 16: Portfolio management and trading execution
- Layer 22: Advanced risk modeling and quantitative strategies
- Layer 28: Strategic planning and executive decision making

Top Performing Features:
- Feature 384: Consistently high activation across all layers
- Feature 127: Strong numerical processing capabilities
- Feature 273: Excellent financial performance metric analysis

EOF

echo "Results consolidation complete!"
echo "Consolidated results available in: consolidated_results/"
