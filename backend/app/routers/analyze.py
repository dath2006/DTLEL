"""
Unified Analysis Pipeline (v3.0 - Optimized)

Combines:
1. SuperAnnotate RoBERTa Large for AI detection (primary)
2. Stylometric N-gram analysis
3. Statistical metrics (perplexity, burstiness) - cached per sentence
4. Plagiarism detection (FAISS + SBERT)

Optimizations:
- Removed redundant ai_detector (RoBERTa Base)
- Perplexity calculated once and reused for flux/burstiness
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.utils.text_extract import extract_text_from_file
from app.utils.chunking import chunk_text
from app.utils.sentence_chunking import sentence_splitter
from app.engine import plagiarism_engine
from app.engine.stylometry import phrase_fingerprint
from app.engine.superannotate_detector import superannotate_detector
from app.models import (
    AnalysisReport, AnalysisSegment, AnalysisMetrics, 
    TextAnalysisRequest, SentenceScore, StylemetryReport
)
import uuid
import numpy as np
from datetime import datetime
from typing import Optional, List, Tuple

router = APIRouter()


# ============================================================================
# STATISTICAL METRICS HELPERS (Optimized)
# These functions use lightweight calculations - perplexity via GPT-2 is moved
# to a lazy-loaded singleton to avoid loading it if not needed.
# ============================================================================

_ppl_model = None
_ppl_tokenizer = None

def _get_perplexity_model():
    """Lazy-load GPT-2 for perplexity calculation."""
    global _ppl_model, _ppl_tokenizer
    if _ppl_model is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from app.config import settings
        print("Loading Perplexity Model (GPT-2)...")
        _ppl_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        _ppl_model = AutoModelForCausalLM.from_pretrained("gpt2").to(settings.DEVICE)
        _ppl_model.eval()
    return _ppl_model, _ppl_tokenizer


def calculate_sentence_perplexities(sentences: List[str]) -> List[float]:
    """
    Calculate perplexity for a list of sentences in one pass.
    Returns list of perplexity values (one per sentence).
    """
    import torch
    from app.config import settings
    
    model, tokenizer = _get_perplexity_model()
    results = []
    
    for sentence in sentences:
        if len(sentence.split()) < 4:
            results.append(0.0)
            continue
            
        try:
            encodings = tokenizer(sentence[:512], return_tensors="pt").to(settings.DEVICE)
            with torch.no_grad():
                outputs = model(encodings.input_ids, labels=encodings.input_ids)
                ppl = torch.exp(outputs.loss).item()
            results.append(ppl if ppl < 10000 else 0.0)  # Cap unreasonable values
        except:
            results.append(0.0)
    
    return results


def calculate_burstiness(sentence_lengths: List[int]) -> float:
    """Calculate burstiness from pre-computed sentence lengths."""
    if len(sentence_lengths) < 2:
        return 0.0
    std_dev = np.std(sentence_lengths)
    mean = np.mean(sentence_lengths)
    return float(std_dev / mean) if mean > 0 else 0.0


def calculate_perplexity_flux(perplexities: List[float]) -> float:
    """Calculate flux from pre-computed perplexity values."""
    valid_ppls = [p for p in perplexities if p > 0]
    if len(valid_ppls) < 2:
        return 0.0
    mean_ppl = np.mean(valid_ppls)
    std_ppl = np.std(valid_ppls)
    return float(std_ppl / mean_ppl) if mean_ppl > 0 else 0.0



def process_analysis(text: str) -> AnalysisReport:
    """
    Core analysis logic with optimized detection pipeline (v3.0).
    
    Pipeline:
    1. Extract sentences and compute perplexity ONCE (cached)
    2. SuperAnnotate RoBERTa Large for AI detection
    3. Stylometric N-gram analysis
    4. Plagiarism detection (FAISS + SBERT)
    5. Compute statistical metrics from cached values
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided")
    
    # ========== 0. Extract Sentences & Create Windows ==========
    sentences = []  # List of (text, start, end) tuples
    windows = []    # List of SentenceWindow objects
    sentence_texts = []
    sentence_lengths = []
    
    try:
        windows, sentences = sentence_splitter.create_windows(text)
        if sentences:
            sentence_texts = [s[0] for s in sentences]
            sentence_lengths = [len(s[0].split()) for s in sentences]
        else:
            # No sentences found, fallback
            sentences = [(text, 0, len(text))]
            sentence_texts = [text]
            sentence_lengths = [len(text.split())]
            windows = []
    except Exception as e:
        print(f"Sentence extraction failed: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: treat whole text as one sentence
        sentences = [(text, 0, len(text))]
        sentence_texts = [text]
        sentence_lengths = [len(text.split())]
        windows = []
    
    # ========== 1. Calculate Perplexity ONCE (Cached for reuse) ==========
    sentence_perplexities = []
    try:
        sentence_perplexities = calculate_sentence_perplexities(sentence_texts)
    except Exception as e:
        print(f"Perplexity calculation failed: {e}")
        sentence_perplexities = [0.0] * len(sentence_texts)
    
    # ========== 2. HYBRID AI Detection (Global + Window) ==========
    # Strategy: Use global score as anchor, window scores for granularity
    
    # A. First, get the GLOBAL document score (this is what works best)
    global_ai_score = 0.0
    try:
        global_ai_score = superannotate_detector.detect(text)
        print(f"Global document AI score: {global_ai_score:.4f}")
    except Exception as e:
        print(f"Global detection failed: {e}")
    
    # B. Then, get WINDOW scores for granular variation
    window_scores = []
    if windows:
        try:
            window_texts = [w.text for w in windows]
            window_scores = superannotate_detector.detect_batch(window_texts)
            print(f"Processed {len(windows)} windows for {len(sentences)} sentences")
            print(f"Window scores range: {min(window_scores):.3f} - {max(window_scores):.3f}")
        except Exception as e:
            print(f"Window detection failed: {e}")
            window_scores = []
    
    # ========== 3. Aggregate with Global Anchoring ==========
    # Each sentence gets: BLEND of global score (anchor) + local window score (variation)
    # This prevents the "low score" problem while maintaining granularity
    
    sentence_scores_list = []
    ai_sentence_count = 0
    
    if window_scores and len(window_scores) == len(windows):
        # Get raw sentence scores from window aggregation
        sentence_results = sentence_splitter.aggregate_to_sentences(
            windows, window_scores, sentences, threshold=0.5
        )
        
        # Blend: 60% global anchor + 40% local window score
        # This ensures high global detection still shows, but with variation
        GLOBAL_WEIGHT = 0.6
        LOCAL_WEIGHT = 0.4
        
        for i, sr in enumerate(sentence_results):
            ppl = sentence_perplexities[i] if i < len(sentence_perplexities) else 0.0
            
            # Hybrid score: anchor to global, vary by local
            hybrid_score = (global_ai_score * GLOBAL_WEIGHT) + (sr.score * LOCAL_WEIGHT)
            
            # Boost if perplexity is suspiciously low (AI-like)
            if ppl > 0 and ppl < 25:
                hybrid_score = min(1.0, hybrid_score + 0.05)
            
            hybrid_score = round(min(1.0, max(0.0, hybrid_score)), 4)
            is_ai = hybrid_score > 0.5
            
            sentence_scores_list.append(SentenceScore(
                text=sr.text,
                index=sr.index,
                start_char=sr.start_char,
                end_char=sr.end_char,
                ai_probability=hybrid_score,
                is_ai_generated=is_ai,
                perplexity=round(ppl, 2),
                window_count=sr.window_count
            ))
            if is_ai:
                ai_sentence_count += 1
    else:
        # Fallback: Only global score available
        for i, (sent_text, start, end) in enumerate(sentences):
            ppl = sentence_perplexities[i] if i < len(sentence_perplexities) else 0.0
            ppl_modifier = 0.05 if (ppl > 0 and ppl < 25) else 0.0
            score = min(1.0, global_ai_score + ppl_modifier)
            is_ai = score > 0.5
            
            sentence_scores_list.append(SentenceScore(
                text=sent_text,
                index=i,
                start_char=start,
                end_char=end,
                ai_probability=round(score, 4),
                is_ai_generated=is_ai,
                perplexity=round(ppl, 2),
                window_count=0
            ))
            if is_ai:
                ai_sentence_count += 1
    
    # Overall score is now the global document score (most reliable)
    superannotate_score_val = global_ai_score



    
    # ========== 4. Stylometric Analysis ==========
    stylometry_report = None
    try:
        style_result = phrase_fingerprint.analyze(text)
        stylometry_report = StylemetryReport(
            total_ai_phrases=style_result.total_ai_phrases,
            unique_phrases=style_result.unique_phrases,
            phrase_density=style_result.phrase_density,
            category_breakdown=style_result.category_breakdown,
            top_phrases=style_result.top_phrases,
            stylometry_score=style_result.stylometry_score
        )
    except Exception as e:
        print(f"Stylometry analysis failed: {e}")
    

    # ========== 3. Plagiarism Detection (Retrieve & Re-Rank) ==========
    plagiarism_segments = []
    total_plagiarism_len = 0
    total_len = max(len(text), 1)
    
    # We use the sentences extracted earlier for granular plagiarism check
    if not sentences:
        # Fallback if no sentences found (rare)
        chunks = chunk_text(text)
        plag_query_texts = chunks
        plag_indices = range(len(chunks))
        plag_start_end = [(0, 0)] * len(chunks) # placeholder
    else:
        plag_query_texts = [s[0] for s in sentences]
        plag_indices = [i for i in range(len(sentences))]
    
    # A. Retrieve Candidates (Top-5 for speed)
    search_results = plagiarism_engine.search(plag_query_texts, top_k=5)
    
    # B. Prepare Batch for Re-Ranking
    # We collect ALL pairs across ALL sentences to run one big batch inference
    all_pairs = []
    pair_metadata = [] # Stores (sentence_index, candidate_result)
    
    for i, results in enumerate(search_results):
        query_text = plag_query_texts[i]
        for res in results:
            cand_text = res['metadata'].get('text', '')
            vector_score = res.get('score', 0.0)
            
            # Early Exit: Skip re-ranking if vector match is weak
            if cand_text and vector_score > 0.35:
                all_pairs.append((query_text[:512], cand_text[:512]))
                pair_metadata.append((i, res))

    # C. Run Batch Inference (One call to GPU/CPU)
    all_cross_scores = []
    if all_pairs:
        try:
            all_cross_scores = plagiarism_engine.compute_cross_scores(all_pairs)
        except Exception as e:
            print(f"Batch Cross-Encoder failed: {e}")
            all_cross_scores = [0.0] * len(all_pairs)

    # D. Process Scores and Assign to Sentences
    # We need to find the BEST match for each sentence
    sentence_best_matches = {} # {sentence_index: (score, match_res)}
    
    for idx, score in enumerate(all_cross_scores):
        sent_idx, match_res = pair_metadata[idx]
        
        current_best = sentence_best_matches.get(sent_idx, (0.0, None))
        if score > current_best[0]:
            sentence_best_matches[sent_idx] = (score, match_res)

    # E. Build Segments
    for i in range(len(plag_query_texts)):
        best_score, best_match = sentence_best_matches.get(i, (0.0, None))
        query_text = plag_query_texts[i]
        
        # Determine if plagiarized
        is_plagiarized = best_score > 0.80
        if is_plagiarized:
            total_plagiarism_len += len(query_text)

        # Map back to AnalysisSegment
        if sentences:
            start = sentences[i][1]
            end = sentences[i][2]
        else:
            start = 0
            end = 0

        # AI Probability for this segment
        ai_prob = 0.0
        if i < len(sentence_scores_list):
            ai_prob = sentence_scores_list[i].ai_probability

        plagiarism_segments.append(AnalysisSegment(
            text=query_text,
            start_index=start,
            end_index=end,
            ai_probability=ai_prob,
            plagiarism_score=best_score if is_plagiarized else 0.0,
            is_plagiarized=is_plagiarized,
            source_id=best_match['id'] if is_plagiarized and best_match else None,
            source_metadata=best_match['metadata'] if is_plagiarized and best_match else None
        ))

    # ========== 5. Calculate Scores ==========
    
    # A. Model Score (SuperAnnotate - already calculated)
    model_score = superannotate_score_val
    
    # B. Plagiarism Score (0-1) - Weighted by length
    plag_percent = (total_plagiarism_len / total_len) * 100
    plagiarism_score = min(1.0, plag_percent / 20)  # 20%+ = max score
    
    # C. Statistical Metrics (Using CACHED perplexity values)
    perplexity = np.mean([p for p in sentence_perplexities if p > 0]) if any(p > 0 for p in sentence_perplexities) else 0.0
    burstiness = calculate_burstiness(sentence_lengths)
    perplexity_flux = calculate_perplexity_flux(sentence_perplexities)
    
    # Use the new segments list
    segments = plagiarism_segments

    
    # C. Stylometry Score (already 0-1)
    stylometry_score_val = stylometry_report.stylometry_score if stylometry_report else 0.0
    phrase_density = stylometry_report.phrase_density if stylometry_report else 0.0
    phrase_count = stylometry_report.total_ai_phrases if stylometry_report else 0
    
    # ========== 6. Ensemble AI Score with Statistical Signals ==========
    # 
    # Statistical signals that indicate AI-generated text:
    # - LOW burstiness (<0.3) = uniform sentence lengths = AI
    # - LOW perplexity (<30) = predictable/repetitive text = AI
    # - LOW perplexity flux (<0.3) = consistently predictable = AI
    # - HIGH stylometry score = many AI phrases
    #
    # We combine these with RoBERTa/Binoculars for robust detection
    
    # A. Statistical penalty for low burstiness (AI texts are uniform)
    # Human text typically has burstiness > 0.5
    if burstiness < 0.3:
        burstiness_penalty = 0.8  # Strong AI signal
    elif burstiness < 0.5:
        burstiness_penalty = 0.4  # Moderate AI signal
    else:
        burstiness_penalty = 0.0  # Human-like
    
    # B. Statistical penalty for Perplexity Flux (AI texts are CONSISTENTLY predictable)
    # Human text has spikes (High Flux). AI text is flat (Low Flux).
    # Typical Flux: Human > 0.5, AI < 0.3
    if perplexity_flux < 0.2:
        flux_penalty = 0.9      # Robotically consistent
    elif perplexity_flux < 0.35:
        flux_penalty = 0.5      # Suspiciously consistent
    else:
        flux_penalty = 0.0      # Human-like variation
    
    # C. Combined statistical signal 
    # Weighted average: Flux is now the strongest specific signal
    statistical_penalty = (burstiness_penalty * 0.4 + flux_penalty * 0.6)
    
    # D. Model-based score (SuperAnnotate - already set as model_score above)
    ensemble_method = "superannotate_stats"

    
    # E. Final AI Score: Use MAX of signals for aggressive detection
    # If any strong signal detected, use it as the score
    # This ensures high stylometry (1.0) or high model score results in high AI score
    
    # Calculate weighted average as baseline
    # Updated weights (v3.0 - SuperAnnotate only): 
    # - Model (50%): SuperAnnotate RoBERTa Large
    # - Statistical (20%): Burstiness/Perplexity Flux
    # - Stylometry (30%): Phrase matching
    weighted_score = (
        model_score * 0.80 +
        statistical_penalty * 0.10 +
        stylometry_score_val * 0.10
    )
    weighted_score = min(1.0, weighted_score)
    
    # Use maximum of: weighted average, stylometry, model score
    # This ensures any strong signal dominates
    # Score Boosting: Map SuperAnnotate's conservative range to full 1.0 scale
    # Boosting starts at 57% (user-configured threshold)
    boosted_sa_score = superannotate_score_val
    if superannotate_score_val > 0.70:
        # Strong AI signal: 0.70 → 0.92, 0.80 → 0.97
        boosted_sa_score = 0.92 + ((superannotate_score_val - 0.70) * 0.5)
    elif superannotate_score_val > 0.57:
        # Moderate AI signal: 0.57 → 0.75, 0.70 → 0.92
        boosted_sa_score = 0.75 + ((superannotate_score_val - 0.57) * 1.31)

    ai_score = max(
        weighted_score, 
        stylometry_score_val,         # Trust strong stylometry fully (1.0)
        model_score,                  # Trust strong model confidence fully (1.0)
        boosted_sa_score              # Trust boosted specialized detector
    )
    ai_score = round(min(1.0, ai_score), 2)
    
    # ========== 7. Legacy Integrity Score ==========
    # For backward compatibility: 1 - max(all scores)
    max_score = max(ai_score, plagiarism_score, stylometry_score_val)
    overall_integrity = round(max(0.0, 1.0 - max_score), 2)
    
    # ========== 7.5. Consistent Sentence Scoring ==========
    # The users complained that "Overall AI is 80% but sentences look Human".
    # This happens because individual sentences are short and hard for RoBERTa to classify,
    # whereas the Ensemble (SuperAnnotate + Stats) sees the full picture.
    # We now BOOST the sentence scores using the Global AI Score to fix this inconsistency.
    
    if sentence_scores_list and ai_score > 0.4:
        # If the document is clearly AI, we shouldn't let sentences stay at "0.01"
        for sent in sentence_scores_list:
            # Blend Local Score (30%) with Global AI Pattern (70%)
            # We heavily weight the global pattern because SuperAnnotate is our source of truth
            original_score = sent.ai_probability
            
            # Contextual Boost: The sentence is part of a document we KNOW is AI.
            # We treat the global score as a "prior probability".
            boosted_score = (original_score * 0.2) + (ai_score * 0.8)
            
            # Update the sentence object
            sent.ai_probability = round(boosted_score, 4)
            sent.is_ai_generated = sent.ai_probability > 0.5
            
            # Recount flagged sentences
            ai_sentence_count = sum(1 for s in sentence_scores_list if s.is_ai_generated)

    # ========== 8. Build Report ==========
    return AnalysisReport(
        report_id=str(uuid.uuid4()),
        timestamp=datetime.now().isoformat(),
        
        # SEPARATE SCORES (main output)
        ai_score=ai_score,
        plagiarism_score=round(plagiarism_score, 2),
        stylometry_score=round(stylometry_score_val, 2),
        
        # Legacy
        overall_integrity_score=overall_integrity,
        
        # Detailed metrics
        metrics=AnalysisMetrics(
            # AI Detection (Ensemble)
            ai_score=ai_score,
            roberta_score=round(model_score, 4),  # Now SuperAnnotate score (legacy field name)
            ensemble_method=ensemble_method,
            
            # Plagiarism
            plagiarism_score=round(plagiarism_score, 2),
            plagiarism_percentage=round(plag_percent, 2),
            
            # Stylometry
            stylometry_score=round(stylometry_score_val, 2),
            ai_phrase_count=phrase_count,
            phrase_density=round(phrase_density, 2),
            
            # Statistical
            burstiness_score=round(burstiness, 2),
            perplexity_avg=round(perplexity, 2),
            perplexity_flux=round(perplexity_flux, 2),
            
            # SuperAnnotate
            superannotate_score=round(superannotate_score_val, 4),
            
            # Sentence stats
            sentence_count=len(sentence_scores_list) if sentence_scores_list else None,
            ai_sentence_count=ai_sentence_count if sentence_scores_list else None
        ),
        segments=segments,
        sentence_scores=sentence_scores_list if sentence_scores_list else None,
        stylometry=stylometry_report
    )


@router.post("/analyze", response_model=AnalysisReport)
async def analyze_document(
    file: UploadFile = File(...)
):
    """
    Analyzes a document (PDF, DOCX, TXT) for Plagiarism and AI Generation.
    
    Features:
    - Sentence-level AI detection using Binoculars (Falcon-7B)
    - Stylometric phrase analysis
    - Legacy chunk-based detection (fallback)
    - FAISS-based plagiarism detection
    """
    try:
        text = await extract_text_from_file(file)
        return process_analysis(text)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/text", response_model=AnalysisReport)
async def analyze_text(
    request: TextAnalysisRequest
):
    """
    Analyzes raw text for Plagiarism and AI Generation.
    """
    try:
        return process_analysis(request.text)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
