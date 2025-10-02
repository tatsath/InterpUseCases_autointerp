#!/bin/bash

# CircuitTracing App Launcher
# This script launches both the Feature Steering app and the Feature Activation Tracker

echo "🚀 Starting CircuitTracing Apps..."

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae

# Check if apps are already running
if pgrep -f "streamlit.*streamlit_feature_steering_app.py" > /dev/null; then
    echo "⚠️  Feature Steering app is already running on port 8501"
else
    echo "🎯 Starting Feature Steering app on port 8501..."
    cd /home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Steering
    streamlit run streamlit_feature_steering_app.py --server.port 8501 &
    sleep 3
fi

if pgrep -f "streamlit.*feature_activation_tracker.py" > /dev/null; then
    echo "⚠️  Feature Activation Tracker is already running on port 8502"
else
    echo "🔍 Starting Feature Activation Tracker on port 8502..."
    cd /home/nvidia/Documents/Hariom/InterpUseCases_autointerp/CircuitTracing
    streamlit run feature_activation_tracker.py --server.port 8502 &
    sleep 3
fi

echo ""
echo "✅ Apps are running!"
echo "🎯 Feature Steering: http://localhost:8501"
echo "🔍 Feature Activation Tracker: http://localhost:8502"
echo ""
echo "Press Ctrl+C to stop all apps"

# Wait for user to stop
wait
