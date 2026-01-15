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


@dataclass
class PhraseMatch:
    """A detected AI phrase in the text."""
    phrase: str
    category: str                # transitions, abstract_nouns, verbs, adjectives, filler_phrases
    count: int
    positions: List[int] = field(default_factory=list)  # Start positions in text


@dataclass
class StylometryReport:
    """Full stylometric analysis report."""
    total_ai_phrases: int        # Total count of AI phrase occurrences
    unique_phrases: int          # Number of distinct AI phrases found
    phrase_density: float        # AI phrases per 100 words (0-100+)
    category_breakdown: Dict[str, int]  # Count per category
    top_phrases: List[Tuple[str, int]]  # Top N most frequent phrases
    flagged_phrases: List[PhraseMatch]  # All detected phrases with details
    stylometry_score: float      # 0.0 (clean) to 1.0 (saturated with AI phrases)


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
    
    def analyze(self, text: str, top_n: int = 10) -> StylometryReport:
        """
        Analyze text for AI-characteristic phrases.
        
        Args:
            text: Input text to analyze
            top_n: Number of top phrases to return
            
        Returns:
            StylometryReport with detailed breakdown
        """
        if not text.strip():
            return StylometryReport(
                total_ai_phrases=0,
                unique_phrases=0,
                phrase_density=0.0,
                category_breakdown={},
                top_phrases=[],
                flagged_phrases=[],
                stylometry_score=0.0
            )
        
        # Count words for density calculation
        word_count = len(text.split())
        
        flagged_phrases = []
        category_counts = defaultdict(int)
        phrase_counts = {}
        
        # Search for each phrase pattern
        for category, phrases in self.phrase_patterns.items():
            for phrase in phrases:
                count, positions = self.count_phrase(text, phrase)
                if count > 0:
                    flagged_phrases.append(PhraseMatch(
                        phrase=phrase,
                        category=category,
                        count=count,
                        positions=positions
                    ))
                    category_counts[category] += count
                    phrase_counts[phrase] = count
        
        # Calculate aggregates
        total_phrases = sum(p.count for p in flagged_phrases)
        unique_phrases = len(flagged_phrases)
        
        # Density: phrases per 100 words
        density = (total_phrases / word_count * 100) if word_count > 0 else 0.0
        
        # Sort by count for top phrases
        top_phrases = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Calculate stylometry score (0-1)
        # Based on density and diversity of AI phrases
        # Thresholds based on research: >2% density is suspicious, >5% is highly suspicious
        density_score = min(1.0, density / 5.0)  # Cap at 5% density
        diversity_score = min(1.0, unique_phrases / 15.0)  # Cap at 15 unique phrases
        stylometry_score = (density_score * 0.7 + diversity_score * 0.3)
        
        return StylometryReport(
            total_ai_phrases=total_phrases,
            unique_phrases=unique_phrases,
            phrase_density=round(density, 2),
            category_breakdown=dict(category_counts),
            top_phrases=top_phrases,
            flagged_phrases=flagged_phrases,
            stylometry_score=round(stylometry_score, 2)
        )
    
    def get_highlighted_text(self, text: str) -> str:
        """
        Return text with AI phrases highlighted using markdown.
        
        Returns:
            Text with **bold** markers around AI phrases
        """
        result = text
        # Track all phrase positions to avoid overlapping replacements
        all_matches = []
        
        for category, phrases in self.phrase_patterns.items():
            for phrase in phrases:
                pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                for match in pattern.finditer(text):
                    all_matches.append((match.start(), match.end(), match.group()))
        
        # Sort by position (reverse order for safe replacement)
        all_matches.sort(key=lambda x: x[0], reverse=True)
        
        # Apply highlights
        for start, end, original in all_matches:
            result = result[:start] + f"**{original}**" + result[end:]
        
        return result


# Singleton instance
phrase_fingerprint = PhraseFingerprint()


def analyze_stylometry(text: str) -> StylometryReport:
    """Convenience function for stylometric analysis."""
    return phrase_fingerprint.analyze(text)
