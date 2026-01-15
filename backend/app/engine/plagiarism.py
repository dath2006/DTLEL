import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Tuple, Dict, Any
from app.config import settings

class PlagiarismEngine:
    def __init__(self):
        self.vector_store_path = "vector_store"
        self.index_path = os.path.join(self.vector_store_path, "plagiarism.index")
        self.metadata_path = os.path.join(self.vector_store_path, "metadata.pkl")
        
        if not os.path.exists(self.vector_store_path):
            os.makedirs(self.vector_store_path)

        print("Loading Plagiarism Model (SBERT)...")
        self.model = SentenceTransformer(settings.PLAGIARISM_MODEL_NAME, device=settings.DEVICE)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        print("Loading Cross-Encoder (Re-Ranker)...")
        # Load the Re-Ranker model recommended for paraphrase detection
        self.cross_encoder = CrossEncoder('cross-encoder/stsb-roberta-large', device=settings.DEVICE)
        
        self.metadata_store: Dict[int, Dict[str, Any]] = {} 
        self.current_id = 0

        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            print("Loading existing vector store...")
            self.load()
        else:
            print("Initializing new FAISS Index...")
            if settings.DEVICE == "cuda":
                self.index = faiss.IndexFlatIP(self.embedding_dim)
            else:
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
        
        # CrossEncoder.predict returns numpy array or float
        scores = self.cross_encoder.predict(pairs)
        
        if isinstance(scores, (int, float)):
            return [float(scores)]
        
        return [float(s) for s in scores]

plagiarism_engine = PlagiarismEngine()
