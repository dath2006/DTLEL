"""
Stylometric N-gram Analysis for AI Detection

Detects overused AI phrases and patterns in text.
Provides interpretable insights beyond probability scores.
"""

import json
import os
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict


from app.models import StylemetryReport, PhraseMatchModel

# Remove local PhraseMatch and StylometryReport dataclasses
# Use PhraseMatchModel instead of PhraseMatch
# Use StylemetryReport instead of StylometryReport


class PhraseFingerprint:
    """
    Detects AI-characteristic phrases and patterns in text.
    
    Why stylometry?
    - AI models have consistent phrase preferences
    - Counting these provides interpretable evidence
    - Useful even when probability models are uncertain
    """
    
    def __init__(self, phrases_path: Optional[str] = None):
        """
        Args:
            phrases_path: Path to JSON file with phrase categories
        """
        if phrases_path is None:
            phrases_path = os.path.join(
                os.path.dirname(__file__), "..", "data", "ai_phrases.json"
            )
        
        self.phrases_path = phrases_path
        self.phrase_patterns: Dict[str, List[str]] = {}
        self._load_phrases()
    
    def _load_phrases(self):
        """Load phrase patterns from JSON file."""
        try:
            with open(self.phrases_path, 'r', encoding='utf-8') as f:
                self.phrase_patterns = json.load(f)
        except FileNotFoundError:
            print(f"Warning: AI phrases file not found at {self.phrases_path}")
            self.phrase_patterns = {
                "transitions": ["Furthermore", "Moreover", "In conclusion"],
                "verbs": ["delve", "underscore", "leverage"],
                "adjectives": ["crucial", "pivotal", "seamless"]
            }
    
    def count_phrase(self, text: str, phrase: str) -> Tuple[int, List[int]]:
        """
        Count occurrences of a phrase in text (case-insensitive).
        
        Returns:
            Tuple of (count, positions) where positions are start indices
        """
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        matches = list(pattern.finditer(text))
        positions = [m.start() for m in matches]
        return len(matches), positions
    
    
    def analyze(self, text: str, top_n: int = 10) -> StylemetryReport:
        """
        Analyze text for AI-characteristic phrases and statistical biomarkers.
        """
        if not text.strip():
            return StylemetryReport(
                total_ai_phrases=0,
                unique_phrases=0,
                phrase_density=0.0,
                category_breakdown={},
                top_phrases=[],
                stylometry_score=0.0,
                readability_score=0.0,
                avg_sentence_length=0.0,
                complex_word_ratio=0.0,
                vocabulary_richness=0.0
            )
        
        # 1. Existing Phase Logic
        word_count = len(text.split())
        
        flagged_phrases = []
        category_counts = defaultdict(int)
        phrase_counts = {}
        
        for category, phrases in self.phrase_patterns.items():
            for phrase in phrases:
                count, positions = self.count_phrase(text, phrase)
                if count > 0:
                    flagged_phrases.append(PhraseMatchModel(
                        phrase=phrase,
                        category=category,
                        count=count,
                        positions=positions
                    ))
                    category_counts[category] += count
                    phrase_counts[phrase] = count
        
        total_phrases = sum(p.count for p in flagged_phrases)
        unique_phrases = len(flagged_phrases)
        density = (total_phrases / word_count * 100) if word_count > 0 else 0.0
        top_phrases = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # 2. Dynamic Statistical Logic
        from app.engine.dynamic_stylometry import stylometry_engine
        dynamic_metrics = {}
        if stylometry_engine:
            dynamic_metrics = stylometry_engine.analyze_text(text)
            
        # 3. Enhanced Scoring
        # Start with phrase-based score
        density_score = min(1.0, density / 5.0)
        diversity_score = min(1.0, unique_phrases / 15.0)
        base_score = (density_score * 0.7 + diversity_score * 0.3)
        
        # Adjust based on dynamic signals
        # Low vocabulary richness (repetitive) -> +0.1
        # Low readability (complex/robotic) -> +0.1
        adjustment = 0.0
        vocab_richness = dynamic_metrics.get("vocabulary_richness", 1.0)
        
        if vocab_richness < 0.4: # Repetitive
             adjustment += 0.15
             
        final_score = min(1.0, base_score + adjustment)
        
        return StylemetryReport(
            total_ai_phrases=total_phrases,
            unique_phrases=unique_phrases,
            phrase_density=round(density, 2),
            category_breakdown=dict(category_counts),
            top_phrases=top_phrases,
            stylometry_score=round(final_score, 2),
            readability_score=dynamic_metrics.get("readability_score", 0.0),
            avg_sentence_length=dynamic_metrics.get("avg_sentence_length", 0.0),
            complex_word_ratio=dynamic_metrics.get("complex_word_ratio", 0.0),
            vocabulary_richness=dynamic_metrics.get("vocabulary_richness", 0.0)
        )
    
    def get_highlighted_text(self, text: str) -> str:
        """
        Return text with AI phrases highlighted using markdown.
        """
        result = text
        all_matches = []
        
        for category, phrases in self.phrase_patterns.items():
            for phrase in phrases:
                pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                for match in pattern.finditer(text):
                    all_matches.append((match.start(), match.end(), match.group()))
        
        all_matches.sort(key=lambda x: x[0], reverse=True)
        
        for start, end, original in all_matches:
            result = result[:start] + f"**{original}**" + result[end:]
        
        return result


# Singleton instance
phrase_fingerprint = PhraseFingerprint()


def analyze_stylometry(text: str) -> StylemetryReport:
    """Convenience function for stylometric analysis."""
    return phrase_fingerprint.analyze(text)
