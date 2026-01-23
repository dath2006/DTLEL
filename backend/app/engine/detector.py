import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import numpy as np
import math
import os
import onnxruntime as ort
from typing import Dict, Any, List, Optional
from app.config import settings

class AIDetector:
    def __init__(self):
        self.device = settings.DEVICE
        self.onnx_path = "onnx_models/ai_detector/model.onnx"
        self.use_onnx = False
        self.tokenizer = AutoTokenizer.from_pretrained(settings.AI_DETECTOR_MODEL_NAME)

        # 1. Try Loading ONNX Model
        if os.path.exists(self.onnx_path):
            print(f"Loading AI Detector (ONNX) from {self.onnx_path}...")
            try:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == "cuda" else ['CPUExecutionProvider']
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.ort_session = ort.InferenceSession(self.onnx_path, sess_options, providers=providers)
                self.use_onnx = True
                print("AI Detector (ONNX) Ready.")
            except Exception as e:
                print(f"Failed to load ONNX model: {e}. Falling back to PyTorch.")
        
        # 2. Fallback to PyTorch
        if not self.use_onnx:
            print(f"Loading AI Detection Model ({settings.AI_DETECTOR_MODEL_NAME})...")
            self.model = AutoModelForSequenceClassification.from_pretrained(settings.AI_DETECTOR_MODEL_NAME).to(self.device)
            if self.device == "cuda":
                self.model = self.model.half()
            self.model.eval()

        print("Loading Perplexity Model (GPT-2)...")
        self.ppl_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.ppl_model = AutoModelForCausalLM.from_pretrained("gpt2").to(self.device)
        if self.device == "cuda":
            self.ppl_model = self.ppl_model.half()
        self.ppl_model.eval()
        print("AI Detector Ready.")

    def detect_probability(self, texts: List[str]) -> List[float]:
        """
        Returns probability of being AI-generated for a list of text chunks.
        """
        probs_list = []
        batch_size = 8
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            if self.use_onnx:
                # --- ONNX Inference ---
                onnx_inputs = {
                    "input_ids": inputs["input_ids"].numpy().astype(np.int64),
                    "attention_mask": inputs["attention_mask"].numpy().astype(np.int64)
                }
                logits = self.ort_session.run(None, onnx_inputs)[0] # Output is [batch, 2]
                
                # Softmax manually using numpy
                # exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                # probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                
                # Torch softmax is faster/easier if we convert back, but let's stay numpy for speed
                logits_tensor = torch.from_numpy(logits)
                probs = torch.softmax(logits_tensor, dim=1)
                
            else:
                # --- PyTorch Inference ---
                inputs = inputs.to(self.device)
                with torch.no_grad():
                    if self.device == "cuda":
                        with torch.amp.autocast('cuda'):
                            logits = self.model(**inputs).logits
                    else:
                        logits = self.model(**inputs).logits
                    probs = torch.softmax(logits, dim=1)
                
            # PirateXX/AI-Content-Detector: Label_0 = Fake (AI), Label_1 = Real (Human)
            batch_probs = probs[:, 0].cpu().tolist()
            probs_list.extend(batch_probs)
            
        return probs_list

    def calculate_perplexity(self, text: str) -> float:
        """
        Calculates perplexity using GPT-2. Lower perplexity = usage of more common patterns (more likely AI).
        """
        encodings = self.ppl_tokenizer(text, return_tensors="pt").to(self.device)
        max_length = self.ppl_model.config.n_positions
        stride = 512
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.ppl_model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        if not nlls:
            return 0.0
            
        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl.item()

    def calculate_perplexity_flux(self, text: str) -> float:
        """
        Calculates the VARIANCE of perplexity across sentences (Flux).
        
        High Flux = Human (Erratic predictability)
        Low Flux = AI (Consistently smoothed predictability)
        """
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
        if len(sentences) < 3:
            return 0.0
            
        sent_ppls = []
        # Calculate perplexity for each sentence individually
        for sentence in sentences:
            # We cap length to avoid OOM on massive sentences
            # and ignore tiny ones
            try:
                ppl = self.calculate_perplexity(sentence[:512])
                if ppl > 0:
                    sent_ppls.append(ppl)
            except Exception:
                continue
                
        if len(sent_ppls) < 2:
            return 0.0
            
        # Calculate Coefficient of Variation (StdDev / Mean)
        # This normalizes the score so high-perplexity texts aren't unfairly weighted
        mean_ppl = np.mean(sent_ppls)
        std_ppl = np.std(sent_ppls)
        
        if mean_ppl == 0:
            return 0.0
            
        return std_ppl / mean_ppl

    def analyze_burstiness(self, text: str) -> float:
        """
        Calculates burstiness based on sentence length variation (std dev).
        High variation = High Burstiness (Human).
        Low variation = Low Burstiness (AI).
        """
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 3]
        if not sentences:
            return 0.0
        
        lengths = [len(s.split()) for s in sentences]
        if len(lengths) < 2:
            return 0.0
            
        std_dev = np.std(lengths)
        mean = np.mean(lengths)
        
        if mean == 0:
            return 0.0
            
        # Coefficient of variation as a proxy for burstiness
        return std_dev / mean

ai_detector = AIDetector()
