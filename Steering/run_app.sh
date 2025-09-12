#!/bin/bash

# Financial LLM Feature Steering App Launcher
# This script launches the Streamlit feature steering application

echo "🎯 Financial LLM Feature Steering App"
echo "======================================"
echo ""
echo "Starting Streamlit application..."
echo "The app will be available at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing requirements..."
    pip install -r requirements.txt
fi

# Launch the Streamlit app
streamlit run streamlit_feature_steering_app.py --server.port 8501 --server.address 0.0.0.0
