"""
Sentence-Based Sliding Window Text Chunking

Replaces fixed character chunking with context-aware sentence grouping.
AI detection models work better when semantic boundaries are preserved.
"""

import spacy
from typing import List, Dict, Tuple, NamedTuple
from dataclasses import dataclass

# Load spaCy model for sentence segmentation
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "textcat"])
except OSError:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "textcat"])


@dataclass
class SentenceWindow:
    """A window of consecutive sentences for analysis."""
    text: str                           # Combined text of sentences in window
    sentence_indices: List[int]         # Indices of sentences in this window
    start_char: int                     # Start character position in original text
    end_char: int                       # End character position in original text


@dataclass 
class SentenceWithScore:
    """A sentence with its aggregated AI detection score."""
    text: str
    index: int
    start_char: int
    end_char: int
    score: float                        # Aggregated score from all windows containing this sentence
    window_count: int                   # How many windows this sentence appeared in
    is_ai_generated: bool


class SentenceWindowSplitter:
    """
    Splits text into sentences and creates overlapping windows for analysis.
    
    Why sliding windows?
    - AI models leave "fingerprints" in long-range coherence
    - Analyzing 3-5 sentences together captures these patterns
    - Overlapping windows allow sentence-level score aggregation
    """
    
    def __init__(self, window_size: int = 4, stride: int = 1, min_sentence_length: int = 10):
        """
        Args:
            window_size: Number of sentences per window (3-5 recommended)
            stride: How many sentences to slide between windows (1 = full overlap)
            min_sentence_length: Minimum characters for a valid sentence
        """
        self.window_size = window_size
        self.stride = stride
        self.min_sentence_length = min_sentence_length
    
    def split_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into sentences using spaCy.
        
        Returns:
            List of (sentence_text, start_char, end_char) tuples
        """
        doc = nlp(text)
        sentences = []
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if len(sent_text) >= self.min_sentence_length:
                sentences.append((
                    sent_text,
                    sent.start_char,
                    sent.end_char
                ))
        
        return sentences
    
    def create_windows(self, text: str) -> Tuple[List[SentenceWindow], List[Tuple[str, int, int]]]:
        """
        Create sliding windows of sentences from text.
        
        Returns:
            Tuple of (windows, sentences) where:
            - windows: List of SentenceWindow objects
            - sentences: Original sentence list for reference
        """
        sentences = self.split_sentences(text)
        
        if not sentences:
            return [], []
        
        # If fewer sentences than window size, create single window
        if len(sentences) <= self.window_size:
            window_text = " ".join([s[0] for s in sentences])
            windows = [SentenceWindow(
                text=window_text,
                sentence_indices=list(range(len(sentences))),
                start_char=sentences[0][1],
                end_char=sentences[-1][2]
            )]
            return windows, sentences
        
        windows = []
        for start_idx in range(0, len(sentences) - self.window_size + 1, self.stride):
            end_idx = start_idx + self.window_size
            window_sentences = sentences[start_idx:end_idx]
            
            window_text = " ".join([s[0] for s in window_sentences])
            windows.append(SentenceWindow(
                text=window_text,
                sentence_indices=list(range(start_idx, end_idx)),
                start_char=window_sentences[0][1],
                end_char=window_sentences[-1][2]
            ))
        
        return windows, sentences
    
    def aggregate_to_sentences(
        self, 
        windows: List[SentenceWindow], 
        window_scores: List[float],
        sentences: List[Tuple[str, int, int]],
        threshold: float = 0.5
    ) -> List[SentenceWithScore]:
        """
        Aggregate window-level scores to sentence-level scores.
        
        Each sentence appears in multiple windows. Its final score is the
        average of all windows it appears in.
        
        Args:
            windows: List of SentenceWindow objects
            window_scores: Scores for each window (higher = more AI-like)
            sentences: Original sentence list
            threshold: Score threshold for AI classification
            
        Returns:
            List of SentenceWithScore with aggregated scores
        """
        # Track scores for each sentence
        sentence_scores: Dict[int, List[float]] = {i: [] for i in range(len(sentences))}
        
        for window, score in zip(windows, window_scores):
            for sent_idx in window.sentence_indices:
                sentence_scores[sent_idx].append(score)
        
        # Aggregate and create result objects
        results = []
        for sent_idx, (sent_text, start_char, end_char) in enumerate(sentences):
            scores = sentence_scores[sent_idx]
            if scores:
                avg_score = sum(scores) / len(scores)
            else:
                avg_score = 0.0
            
            results.append(SentenceWithScore(
                text=sent_text,
                index=sent_idx,
                start_char=start_char,
                end_char=end_char,
                score=round(avg_score, 4),
                window_count=len(scores),
                is_ai_generated=avg_score > threshold
            ))
        
        return results


# Singleton instance with default config
sentence_splitter = SentenceWindowSplitter(window_size=6, stride=1)


def split_into_sentence_windows(text: str, window_size: int = 6, stride: int = 1) -> Tuple[List[SentenceWindow], List[Tuple[str, int, int]]]:
    """
    Convenience function to split text into sentence windows.
    
    Args:
        text: Input text to split
        window_size: Number of sentences per window
        stride: Sentences to slide between windows
        
    Returns:
        Tuple of (windows, sentences)
    """
    splitter = SentenceWindowSplitter(window_size=window_size, stride=stride)
    return splitter.create_windows(text)
