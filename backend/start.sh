#!/bin/bash
set -e

echo "Starting DTLEL Backend..."

# Check if ONNX models exist
if [ ! -f "onnx_models/ai_detector/model.onnx" ]; then
    echo "ONNX models not found. Converting PyTorch models to ONNX..."
    python3 export_onnx.py
    echo "ONNX conversion complete!"
else
    echo "ONNX models found. Skipping conversion."
fi

# Start the application
exec uvicorn main:app --host 0.0.0.0 --port 8000
