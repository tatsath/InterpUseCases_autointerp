#!/bin/bash

# Financial Sentiment Prediction Script
# Uses the trained probe to predict financial sentiment probabilities

SCRIPT_DIR="/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Probes"
PYTHON_SCRIPT="$SCRIPT_DIR/1.get_financial_probabilities.py"

if [ $# -eq 0 ]; then
    echo "Usage: $0 'Your financial news text here'"
    echo ""
    echo "Examples:"
    echo "  $0 'Apple stock surged 8% after strong earnings'"
    echo "  $0 'Tesla shares dropped 3% following delays'"
    echo "  $0 'Microsoft stock remained stable today'"
    echo ""
    echo "The script will show probabilities for all three classes:"
    echo "  - Down (0): Negative sentiment"
    echo "  - Neutral (1): Neutral sentiment" 
    echo "  - Up (2): Positive sentiment"
    exit 1
fi

# Join all arguments as a single text
TEXT="$*"

echo "Analyzing financial sentiment for: '$TEXT'"
echo ""

# Run the Python script
cd "$SCRIPT_DIR"
python "$PYTHON_SCRIPT" "$TEXT"
