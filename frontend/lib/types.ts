export interface IngestResponse {
  id: number;
  message: string;
  chunk_count: number;
}

// ========== Stylometry Types ==========

export interface PhraseMatch {
  phrase: string;
  category: string;
  count: number;
  positions: number[];
}

export interface StylemetryReport {
  total_ai_phrases: number;
  unique_phrases: number;
  phrase_density: number;
  category_breakdown: Record<string, number>;
  top_phrases: [string, number][];
  stylometry_score: number;
  
  // Dynamic Metrics
  readability_score: number;
  avg_sentence_length: number;
  complex_word_ratio: number;
  vocabulary_richness: number;
}

// ========== Sentence-Level Types ==========

export interface SentenceScore {
  text: string;
  index: number;
  start_char: number;
  end_char: number;
  ai_probability: number;
  is_ai_generated: boolean;
  perplexity?: number;
  window_count: number;
}

// ========== Analysis Types ==========

export interface AnalysisSegment {
  text: string;
  start_index: number;
  end_index: number;
  ai_probability: number;
  plagiarism_score: number;
  is_plagiarized: boolean;
  source_id?: number;
  source_metadata?: Record<string, any> | null;
}

export interface AnalysisMetrics {
  // AI Detection (Ensemble)
  ai_score: number;                      // Final AI probability (0-1)
  roberta_score?: number | null;         // RoBERTa classifier score
  ensemble_method: string;               // "roberta_only", "ensemble_avg", etc.
  
  // Plagiarism
  plagiarism_score: number;              // 0-1 score
  plagiarism_percentage: number;         // Percentage flagged
  
  // Stylometry
  stylometry_score: number;              // 0-1 score
  ai_phrase_count: number;
  phrase_density: number;
  
  // Statistical
  burstiness_score: number;
  perplexity_avg: number;
  perplexity_flux?: number;
  
  // Sentence stats
  sentence_count?: number | null;
  ai_sentence_count?: number | null;
}

export interface AnalysisReport {
  report_id: string;
  timestamp: string;
  
  // SEPARATE SCORES (Main Output)
  ai_score: number;                      // 0-1, HIGHER = more AI
  plagiarism_score: number;              // 0-1, HIGHER = more plagiarized
  stylometry_score: number;              // 0-1, HIGHER = more AI phrases
  
  // Legacy
  overall_integrity_score: number;       // For backward compatibility
  
  // Details
  metrics: AnalysisMetrics;
  segments: AnalysisSegment[];
  sentence_scores?: SentenceScore[] | null;
  stylometry?: StylemetryReport | null;
}

