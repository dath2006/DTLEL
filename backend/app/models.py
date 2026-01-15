from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class IngestResponse(BaseModel):
    id: int
    message: str
    chunk_count: int


# ========== Stylometry Models ==========

class PhraseMatchModel(BaseModel):
    """A detected AI phrase in the text."""
    phrase: str
    category: str
    count: int
    positions: List[int] = []


class StylemetryReport(BaseModel):
    """Stylometric analysis of AI-characteristic phrases."""
    total_ai_phrases: int
    unique_phrases: int
    phrase_density: float           # AI phrases per 100 words
    category_breakdown: Dict[str, int]
    top_phrases: List[tuple]        # [(phrase, count), ...]
    stylometry_score: float         # 0.0 (clean) to 1.0 (saturated)


# ========== Sentence-Level Models ==========

class SentenceScore(BaseModel):
    """AI detection score for a single sentence."""
    text: str
    index: int
    start_char: int
    end_char: int
    ai_probability: float           # 0.0 (human) to 1.0 (AI)
    is_ai_generated: bool
    perplexity: Optional[float] = None # Local perplexity score
    window_count: int               # How many windows this sentence appeared in


# ========== Analysis Models (Updated) ==========

class AnalysisSegment(BaseModel):
    """Legacy segment model for backward compatibility."""
    text: str
    start_index: int
    end_index: int
    ai_probability: float
    plagiarism_score: float
    is_plagiarized: bool
    source_id: Optional[int] = None
    source_metadata: Optional[Dict[str, Any]] = None


class AnalysisMetrics(BaseModel):
    """Aggregated metrics from all detection methods."""
    
    # ========== AI Detection (Ensemble) ==========
    ai_score: float                           # Final AI probability (0-1), HIGHER = more AI
    roberta_score: Optional[float] = None     # RoBERTa classifier probability
    ensemble_method: str = "roberta_only"     # "roberta_only", "ensemble_avg", "ensemble_max"
    
    # ========== Plagiarism Detection ==========
    plagiarism_score: float                   # 0-1, HIGHER = more plagiarized
    plagiarism_percentage: float              # Percentage of text flagged
    
    # ========== Stylometry Detection ==========
    stylometry_score: float                   # 0-1, HIGHER = more AI phrases
    ai_phrase_count: int = 0
    phrase_density: float = 0.0               # Phrases per 100 words
    
    # ========== Statistical Metrics ==========
    burstiness_score: float                   # Higher = more human-like variation
    perplexity_avg: float                     # Higher = less predictable (more human)
    perplexity_flux: float = 0.0              # Higher = more human-like variation (Strongest signal)
    
    # ========== SuperAnnotate ==========
    superannotate_score: Optional[float] = None  # Fine-tuned RoBERTa Large from SuperAnnotate
    
    # ========== Sentence-level stats ==========
    sentence_count: Optional[int] = None
    ai_sentence_count: Optional[int] = None


class TextAnalysisRequest(BaseModel):
    text: str


class AnalysisReport(BaseModel):
    """Complete analysis report with SEPARATE scores for each detection method."""
    report_id: str
    timestamp: str
    
    # ========== SEPARATE SCORES (Main Output) ==========
    ai_score: float                 # 0-1, HIGHER = more likely AI-generated
    plagiarism_score: float         # 0-1, HIGHER = more plagiarized
    stylometry_score: float         # 0-1, HIGHER = more AI phrases
    
    # Legacy field (kept for backward compatibility, now just mirrors ai_score inverted)
    overall_integrity_score: float  # 0-1, LOWER = more suspicious (1 - max(ai, plag, style))
    
    # Detailed metrics
    metrics: AnalysisMetrics
    
    # Segment data
    segments: List[AnalysisSegment]
    sentence_scores: Optional[List[SentenceScore]] = None
    stylometry: Optional[StylemetryReport] = None
