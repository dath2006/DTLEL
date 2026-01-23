"""
ONNX Export Script for All Backend Models

Exports:
1. AI Detector (RoBERTa) - Done
2. SBERT (Sentence Embeddings)
3. Cross-Encoder (Re-Ranker)
4. SuperAnnotate (RoBERTa Large)
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaTokenizer, RobertaModel
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
from pathlib import Path
import numpy as np

def export_sequence_classifier(model_name: str, output_path: str):
    """Export HuggingFace Sequence Classification models (RoBERTa-based)."""
    print(f"Exporting {model_name} to {output_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    text = "This is a sample sentence to trace the graph."
    inputs = tokenizer(text, return_tensors="pt")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        (inputs['input_ids'], inputs['attention_mask']),
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size'}
        }
    )
    print(f"Successfully exported to {output_path}")


def export_sbert(model_name: str, output_path: str):
    """Export SentenceTransformer to ONNX."""
    print(f"Exporting SBERT ({model_name}) to {output_path}...")
    
    model = SentenceTransformer(model_name)
    
    # Get the underlying transformer model
    transformer = model[0].auto_model
    tokenizer = model[0].tokenizer
    
    text = "This is a sample sentence."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # SBERT outputs token embeddings, we need pooling after
    class SBERTWrapper(torch.nn.Module):
        def __init__(self, transformer):
            super().__init__()
            self.transformer = transformer
        
        def forward(self, input_ids, attention_mask):
            outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
            # Mean pooling
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
    
    wrapped_model = SBERTWrapper(transformer)
    wrapped_model.eval()
    
    torch.onnx.export(
        wrapped_model,
        (inputs['input_ids'], inputs['attention_mask']),
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask'],
        output_names=['embeddings'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'embeddings': {0: 'batch_size'}
        }
    )
    print(f"Successfully exported SBERT to {output_path}")
    # Also save tokenizer for later use
    tokenizer.save_pretrained(output_path.parent / "tokenizer")


def export_cross_encoder(model_name: str, output_path: str):
    """Export CrossEncoder to ONNX."""
    print(f"Exporting Cross-Encoder ({model_name}) to {output_path}...")
    
    cross_encoder = CrossEncoder(model_name)
    model = cross_encoder.model
    tokenizer = cross_encoder.tokenizer
    
    model.eval()
    
    # Cross-encoder takes sentence pairs
    text_pair = ["This is sentence A.", "This is sentence B."]
    inputs = tokenizer(text_pair[0], text_pair[1], return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        (inputs['input_ids'], inputs['attention_mask']),
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size'}
        }
    )
    print(f"Successfully exported Cross-Encoder to {output_path}")
    tokenizer.save_pretrained(output_path.parent / "tokenizer")


if __name__ == "__main__":
    # 1. AI Detector (Already done, but included for completeness)
    export_sequence_classifier(
        "PirateXX/AI-Content-Detector", 
        "onnx_models/ai_detector/model.onnx"
    )
    
    # 2. SBERT for Plagiarism Embeddings
    export_sbert(
        "all-mpnet-base-v2",
        "onnx_models/sbert/model.onnx"
    )
    
    # 3. Cross-Encoder for Re-Ranking
    export_cross_encoder(
        "cross-encoder/stsb-roberta-large",
        "onnx_models/cross_encoder/model.onnx"
    )
    
    # 4. SuperAnnotate (uses custom class, handled separately)
    print("\n[INFO] SuperAnnotate model requires special handling due to custom architecture.")
    print("[INFO] It's recommended to keep SuperAnnotate in PyTorch with FP16 for now.")
    print("[INFO] For Triton deployment, export via `optimum-cli` with custom config.")
    
    print("\n=== Export Complete ===")
