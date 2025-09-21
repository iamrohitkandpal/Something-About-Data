#!/bin/bash

# Retail Demand Forecasting App Launch Script

echo "🚀 Starting Retail Demand Forecasting Application..."

# Check if running in development or production
if [ "$1" = "dev" ]; then
    echo "🔧 Running in development mode"
    streamlit run main.py --server.runOnSave=true
elif [ "$1" = "prod" ]; then
    echo "🌐 Running in production mode"
    streamlit run main.py --server.headless=true --server.port=8501 --server.address=0.0.0.0
else
    echo "📱 Running in default mode"
    streamlit run main.py
fi