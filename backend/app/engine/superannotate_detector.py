"""
SuperAnnotate AI Detector Adapter

Integrates the SuperAnnotate `generated_text_detector` (based on RoBERTa Large).
Ported from https://github.com/superannotateai/generated_text_detector

Features:
- Robust fine-tuned RoBERTa Large model
- Handling of long texts via chunking
- Separation of code blocks (which are usually false positives)
"""

import torch
import torch.nn.functional as F
import re
from typing import List, Tuple, Optional
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
from huggingface_hub import PyTorchModelHubMixin
from app.config import settings

class RobertaClassifier(nn.Module, PyTorchModelHubMixin):
    """
    Custom Roberta Classifier matching SuperAnnotate architecture.
    Required for loading their checkpoints correctly.
    """
    def __init__(self, config: dict):
        super().__init__()
        # Config from HF contains keys like 'pretrain_checkpoint'
        self.roberta = RobertaModel.from_pretrained(config["pretrain_checkpoint"], add_pooling_layer=False)
        self.dropout = nn.Dropout(config.get("classifier_dropout", 0.1))
        # The checkpoint maps hidden_size -> num_labels directly
        self.dense = nn.Linear(self.roberta.config.hidden_size, config.get("num_labels", 1))

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        # Take <s> token (CLS)
        x = outputs[0][:, 0, :] 
        x = self.dropout(x)
        logits = self.dense(x)
        return float('nan'), logits # mocking loss return signature if needed, or just return logits


# Preprocessing function ported from utils/preprocessing.py
def preprocessing_text(text: str) -> str:
    """Preprocess text for model input."""
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

class SuperAnnotateDetector:
    """
    Adapter for SuperAnnotate's AI Detection model.
    """
    
    def __init__(self, model_name: str = "SuperAnnotate/ai-detector", lazy_load: bool = True):
        self.model_name = model_name
        self.device = None
        self.tokenizer = None
        self.model = None
        self._initialized = False
        self.max_len = 512
        self.code_default_score = 0.5
        self.code_block_pattern = re.compile(r"```(\w+)?\s*([\s\S]*?)\s*```")
        
        if not lazy_load:
            self._initialize()

    def _initialize(self):
        """Load the model and tokenizer."""
        if self._initialized:
            return

        print(f"Loading SuperAnnotate Detector ({self.model_name})...")
        print("Note: This uses RoBERTa Large (~1.5GB) and requires GPU memory.")
        
        self.device = settings.DEVICE
        
        try:
            self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name, do_lower_case=True)
            # Use the custom classifier class
            self.model = RobertaClassifier.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Optimization for GPU
            if self.device == 'cuda':
                self.model = self.model.half()
                # self.model = torch.compile(self.model) # Optional: compile for speed
                
            self._initialized = True
            print("SuperAnnotate Detector Ready.")
        except Exception as e:
            print(f"Failed to load SuperAnnotate detector: {e}")
            raise

    def __split_by_chunks(self, text: str) -> List[str]:
        """Split text into chunks that fit within model max_len."""
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) < self.max_len - 2: # -2 for special tokens
            return [text]

        chunks = []
        # Simple windowing for now, could be improved with nltk sentence splitting
        # Using a safer char-based sliding window to avoid tokenization issues in loop
        words = text.split()
        current_chunk = []
        current_len = 0
        
        for word in words:
            # Approx token count (word + space)
            word_len = len(self.tokenizer.encode(word, add_special_tokens=False))
            if current_len + word_len < self.max_len - 2:
                current_chunk.append(word)
                current_len += word_len
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_len = word_len
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def __predict_chunks(self, texts: List[str]) -> List[float]:
        """Run inference on a batch of text chunks."""
        if not texts:
            return []
            
        inputs = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='longest',
            truncation=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            # Custom forward returns (loss, logits) or just logits? 
            # Original code returns (loss, logits) if cls_output is None/False.
            # My simplified forward returns (nan, logits)
            _, logits = self.model(**inputs)
            
        probas = torch.sigmoid(logits).squeeze(1)
        return probas.tolist()

    def _split_text_and_code(self, text: str) -> Tuple[str, str]:
        """Separate code blocks from standard text."""
        code_blocks = self.code_block_pattern.findall(text)
        code_blocks = [code for lang, code in code_blocks]
        code_content = "\n\n".join(code_blocks)

        # Remove code blocks from text
        clean_text = self.code_block_pattern.sub("", text)
        clean_text = re.sub(r"\n+", "\n", clean_text).strip()

        return clean_text, code_content

    def detect(self, text: str) -> float:
        """
        Detect if text is AI generated.
        Returns score 0.0 (Human) to 1.0 (AI).
        """
        if not self._initialized:
            self._initialize()
            
        clean_text, code_content = self._split_text_and_code(text)
        
        scores = []
        weights = []
        
        # Process Text
        if clean_text.strip():
            # Preprocess
            processed_text = preprocessing_text(clean_text)
            chunks = self.__split_by_chunks(processed_text)
            chunk_scores = self.__predict_chunks(chunks)
            
            for chunk, score in zip(chunks, chunk_scores):
                scores.append(score)
                weights.append(len(chunk))
        
        # Process Code (assign default score)
        if code_content.strip():
            scores.append(self.code_default_score)
            weights.append(len(code_content))
            
        if not scores:
            return 0.0
            
        # Weighted mean
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
            
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        return round(weighted_score, 4)

    @property
    def is_available(self) -> bool:
        return self._initialized

# Singleton
_superannotate_detector: Optional[SuperAnnotateDetector] = None

def get_superannotate_detector() -> SuperAnnotateDetector:
    global _superannotate_detector
    if _superannotate_detector is None:
        _superannotate_detector = SuperAnnotateDetector(lazy_load=True)
    return _superannotate_detector

superannotate_detector = get_superannotate_detector()
