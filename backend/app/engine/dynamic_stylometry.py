import spacy
from typing import Dict, Tuple
from collections import Counter
import textstat

class StylometryEngine:
    """
    Dynamic stylometry engine that calculates linguistic biomarkers using Spacy.
    
    Metrics:
    - Readability (Flesch Reading Ease)
    - Sentence Length Metrics
    - Vocabulary Richness (Type-Token Ratio)
    - POS Tag Distribution
    """
    
    def __init__(self, model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model)
        except OSError:
            print(f"Warning: Spacy model '{model}' not found. Downloading...")
            from spacy.cli import download
            download(model)
            self.nlp = spacy.load(model)

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze text and return statistical metrics.
        """
        if not text.strip():
            return {
                "readability_score": 0.0,
                "avg_sentence_length": 0.0,
                "complex_word_ratio": 0.0,
                "vocabulary_richness": 0.0
            }

        doc = self.nlp(text)
        
        # 1. Readability
        try:
            readability = textstat.flesch_reading_ease(text)
        except:
            readability = 0.0
            
        # 2. Sentence Length
        sentences = list(doc.sents)
        avg_sent_len = sum(len(s) for s in sentences) / len(sentences) if sentences else 0
        
        # 3. Vocabulary Richness (Type-Token Ratio)
        words = [token.text.lower() for token in doc if token.is_alpha]
        unique_words = set(words)
        ttr = len(unique_words) / len(words) if words else 0.0
        
        # 4. Complex Words (Polysyllabic)
        complex_count = textstat.difficult_words(text)
        complex_ratio = complex_count / len(words) if words else 0.0

        return {
            "readability_score": float(readability),
            "avg_sentence_length": float(avg_sent_len),
            "complex_word_ratio": float(complex_ratio),
            "vocabulary_richness": float(ttr)
        }

# Singleton
try:
    stylometry_engine = StylometryEngine()
except Exception as e:
    print(f"Failed to initialize StylometryEngine: {e}")
    stylometry_engine = None
