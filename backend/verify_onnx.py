import time
import torch
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os

MODEL_NAME = "PirateXX/AI-Content-Detector"
ONNX_PATH = "onnx_models/ai_detector/model.onnx"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def benchmark():
    print(f"Benchmarking on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    text = "Artificial Intelligence has revolutionized the way we process information." * 50 # 500+ tokens
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    # 1. PyTorch Benchmark
    print("\n--- PyTorch (FP16) ---")
    pt_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
    if DEVICE == "cuda":
        pt_model = pt_model.half()
    pt_model.eval()
    
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
             pt_model(input_ids, attention_mask)
             
    start = time.time()
    for _ in range(50):
        with torch.no_grad():
            pt_model(input_ids, attention_mask)
    pt_time = (time.time() - start) / 50 * 1000
    print(f"Avg Inference Time: {pt_time:.2f} ms")

    # 2. ONNX Benchmark
    print(f"\n--- ONNX Runtime ({'CUDA' if 'CUDAExecutionProvider' in ort.get_available_providers() else 'CPU'}) ---")
    if not os.path.exists(ONNX_PATH):
        print(f"Error: {ONNX_PATH} not found. Run export_onnx.py first.")
        return

    providers = ['CUDAExecutionProvider'] if DEVICE == "cuda" else ['CPUExecutionProvider']
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(ONNX_PATH, sess_options, providers=providers)
    
    onnx_inputs = {
        "input_ids": inputs["input_ids"].cpu().numpy(),
        "attention_mask": inputs["attention_mask"].cpu().numpy()
    }
    
    # Warmup
    for _ in range(10):
        session.run(None, onnx_inputs)
        
    start = time.time()
    for _ in range(50):
        session.run(None, onnx_inputs)
    onnx_time = (time.time() - start) / 50 * 1000
    print(f"Avg Inference Time: {onnx_time:.2f} ms")
    
    print(f"\nSpeedup: {pt_time / onnx_time:.2f}x")

if __name__ == "__main__":
    benchmark()
