import faiss
import numpy as np
import pickle
import os
import onnxruntime as ort
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer
from typing import List, Tuple, Dict, Any, Optional
from app.config import settings

class PlagiarismEngine:
    def __init__(self):
        self.vector_store_path = "vector_store"
        self.index_path = os.path.join(self.vector_store_path, "plagiarism.index")
        self.metadata_path = os.path.join(self.vector_store_path, "metadata.pkl")
        
        # ONNX paths
        self.sbert_onnx_path = "onnx_models/sbert/model.onnx"
        self.cross_encoder_onnx_path = "onnx_models/cross_encoder/model.onnx"
        
        self.use_onnx_sbert = False
        self.use_onnx_cross_encoder = False
        
        if not os.path.exists(self.vector_store_path):
            os.makedirs(self.vector_store_path)

        # ========== SBERT Loading ==========
        if os.path.exists(self.sbert_onnx_path):
            print(f"Loading SBERT (ONNX) from {self.sbert_onnx_path}...")
            try:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if settings.DEVICE == "cuda" else ['CPUExecutionProvider']
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.sbert_session = ort.InferenceSession(self.sbert_onnx_path, sess_options, providers=providers)
                self.sbert_tokenizer = AutoTokenizer.from_pretrained("onnx_models/sbert/tokenizer")
                self.use_onnx_sbert = True
                self.embedding_dim = 768  # all-mpnet-base-v2 dimension
                print("SBERT (ONNX) Ready.")
            except Exception as e:
                print(f"Failed to load SBERT ONNX: {e}. Falling back to PyTorch.")
        
        if not self.use_onnx_sbert:
            print("Loading Plagiarism Model (SBERT - PyTorch)...")
            self.model = SentenceTransformer(settings.PLAGIARISM_MODEL_NAME, device=settings.DEVICE)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            if settings.DEVICE == "cuda":
                self.model.half()

        # ========== Cross-Encoder Loading ==========
        if os.path.exists(self.cross_encoder_onnx_path):
            print(f"Loading Cross-Encoder (ONNX) from {self.cross_encoder_onnx_path}...")
            try:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if settings.DEVICE == "cuda" else ['CPUExecutionProvider']
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.cross_encoder_session = ort.InferenceSession(self.cross_encoder_onnx_path, sess_options, providers=providers)
                self.cross_encoder_tokenizer = AutoTokenizer.from_pretrained("onnx_models/cross_encoder/tokenizer")
                self.use_onnx_cross_encoder = True
                print("Cross-Encoder (ONNX) Ready.")
            except Exception as e:
                print(f"Failed to load Cross-Encoder ONNX: {e}. Falling back to PyTorch.")
        
        if not self.use_onnx_cross_encoder:
            print("Loading Cross-Encoder (Re-Ranker - PyTorch)...")
            self.cross_encoder = CrossEncoder('cross-encoder/stsb-roberta-large', device=settings.DEVICE)
            if settings.DEVICE == "cuda":
                self.cross_encoder.model.half()
        
        self.metadata_store: Dict[int, Dict[str, Any]] = {} 
        self.current_id = 0

        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            print("Loading existing vector store...")
            self.load()
        else:
            print("Initializing new FAISS Index...")
            self.index = faiss.IndexFlatIP(self.embedding_dim)

        print("Plagiarism Engine Ready.")

    def save(self):
        print("Saving vector store to disk...")
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump({
                "metadata": self.metadata_store,
                "current_id": self.current_id
            }, f)

    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.metadata_store = data["metadata"]
            self.current_id = data["current_id"]

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.use_onnx_sbert:
            # ONNX Inference with Batching
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                inputs = self.sbert_tokenizer(batch_texts, return_tensors="np", padding=True, truncation=True, max_length=512)
                onnx_inputs = {
                    "input_ids": inputs["input_ids"].astype(np.int64),
                    "attention_mask": inputs["attention_mask"].astype(np.int64)
                }
                batch_embeddings = self.sbert_session.run(None, onnx_inputs)[0]
                all_embeddings.append(batch_embeddings)
            
            if not all_embeddings:
                return np.array([])
            
            embeddings = np.vstack(all_embeddings)
            
            # Normalize for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-9)
            return embeddings
        else:
            # PyTorch Fallback
            embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return embeddings

    def add_to_index(self, texts: List[str], metadatas: List[Dict[str, Any]]):
        if not texts:
            return
        
        vectors = self.encode(texts)
        self.index.add(vectors)
        
        for meta in metadatas:
            self.metadata_store[self.current_id] = meta
            self.current_id += 1
        
        # Auto-save after write
        self.save()

    def search(self, query_texts: List[str], top_k: int = 5) -> List[List[Dict[str, Any]]]:
        if self.index.ntotal == 0:
            return [[] for _ in query_texts]
            
        vectors = self.encode(query_texts)
        distances, indices = self.index.search(vectors, top_k)
        
        results = []
        for i, row in enumerate(indices):
            query_results = []
            for j, idx in enumerate(row):
                if idx != -1 and idx in self.metadata_store:
                    meta = self.metadata_store[idx]
                    score = float(distances[i][j])
                    query_results.append({
                        "id": int(idx), # Ensure native int
                        "score": score, # Cosine similarity
                        "metadata": meta
                    })
            results.append(query_results)
        return results

    def delete_document(self, filename: str) -> int:
        """
        Soft-deletes a document by removing its chunks from metadata.
        Returns the number of chunks deleted.
        """
        ids_to_remove = []
        for idx, meta in self.metadata_store.items():
            if meta.get("source") == filename:
                ids_to_remove.append(idx)
        
        for idx in ids_to_remove:
            del self.metadata_store[idx]
            
        if ids_to_remove:
            print(f"Deleted {len(ids_to_remove)} chunks for {filename} from metadata.")
            self.save()
            
        return len(ids_to_remove)

    def clear_index(self):
        """
        Resets the entire vector store and metadata.
        """
        print("Clearing Plagiarism Engine Index...")
        self.index.reset()
        self.metadata_store = {}
        self.current_id = 0
        self.save()
        print("Index cleared.")

    def compute_cross_scores(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Computes the Cross-Encoder similarity scores for a list of (query, candidate) pairs.
        Returns: List of scores (0.0 to 1.0).
        """
        if not pairs:
            return []
        
        if self.use_onnx_cross_encoder:
            # ONNX Inference
            scores = []
            for query, candidate in pairs:
                inputs = self.cross_encoder_tokenizer(query, candidate, return_tensors="np", padding=True, truncation=True, max_length=512)
                onnx_inputs = {
                    "input_ids": inputs["input_ids"].astype(np.int64),
                    "attention_mask": inputs["attention_mask"].astype(np.int64)
                }
                logits = self.cross_encoder_session.run(None, onnx_inputs)[0]
                # Sigmoid for probability
                score = 1 / (1 + np.exp(-logits[0][0]))
                scores.append(float(score))
            return scores
        else:
            # PyTorch Fallback
            scores = self.cross_encoder.predict(pairs)
            
            if isinstance(scores, (int, float)):
                return [float(scores)]
            
            return [float(s) for s in scores]

plagiarism_engine = PlagiarismEngine()
