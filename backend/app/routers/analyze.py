"""
Unified Analysis Pipeline (v2.0)

Combines:
1. Sentence-based sliding window AI detection (Binoculars)
2. Stylometric N-gram analysis
3. Legacy chunk-based RoBERTa fallback
4. Plagiarism detection (FAISS + SBERT)
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.utils.text_extract import extract_text_from_file
from app.utils.chunking import chunk_text
from app.utils.sentence_chunking import sentence_splitter
from app.engine import plagiarism_engine, ai_detector
from app.engine.stylometry import phrase_fingerprint
from app.models import (
    AnalysisReport, AnalysisSegment, AnalysisMetrics, 
    TextAnalysisRequest, SentenceScore, StylemetryReport
)
import uuid
from datetime import datetime
from typing import Optional

router = APIRouter()





def process_analysis(text: str) -> AnalysisReport:
    """
    Core analysis logic with upgraded detection pipeline.
    
    Pipeline:
    1. Sentence-based sliding window analysis (RoBERTa)
    2. Stylometric N-gram analysis
    3. Legacy chunk-based detection (fallback + plagiarism)
    4. Aggregate all signals into final report
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided")
    
    # ========== 1. Sentence-Based AI Detection ==========
    # Use RoBERTa for sentence-level scoring (lightweight fallback)
    sentence_scores_list = []
    ai_sentence_count = 0
    
    try:
        windows, sentences = sentence_splitter.create_windows(text)
        
        if windows and sentences:
            # Score WINDOWS (groups of sentences) with RoBERTa/Binoculars
            window_texts = [w.text for w in windows]
            window_scores = ai_detector.detect_probability(window_texts)
            
            # Aggregate window scores back to individual sentences
            sentence_results = sentence_splitter.aggregate_to_sentences(
                windows, window_scores, sentences, threshold=0.5
            )
            
            # Convert to response model
            for sr in sentence_results:
                # Calculate local perplexity for Granular Insights
                local_ppl = 0.0
                try:
                    if len(sr.text.split()) > 3: # Only distinct check for meaningful sentences
                         local_ppl = ai_detector.calculate_perplexity(sr.text)
                except:
                    pass

                sentence_scores_list.append(SentenceScore(
                    text=sr.text,
                    index=sr.index,
                    start_char=sr.start_char,
                    end_char=sr.end_char,
                    ai_probability=sr.score,
                    is_ai_generated=sr.is_ai_generated,
                    perplexity=round(local_ppl, 2),
                    window_count=sr.window_count
                ))
                if sr.is_ai_generated:
                    ai_sentence_count += 1
                
    except Exception as e:
        print(f"Sentence analysis failed: {e}")
        import traceback
        traceback.print_exc()

    
    # ========== 2. Stylometric Analysis ==========
    stylometry_report = None
    try:
        style_result = phrase_fingerprint.analyze(text)
        stylometry_report = StylemetryReport(
            total_ai_phrases=style_result.total_ai_phrases,
            unique_phrases=style_result.unique_phrases,
            phrase_density=style_result.phrase_density,
            category_breakdown=style_result.category_breakdown,
            top_phrases=style_result.top_phrases,
            stylometry_score=style_result.stylometry_score,
            readability_score=style_result.readability_score,
            avg_sentence_length=style_result.avg_sentence_length,
            complex_word_ratio=style_result.complex_word_ratio,
            vocabulary_richness=style_result.vocabulary_richness
        )
    except Exception as e:
        print(f"Stylometry analysis failed: {e}")
        
    # ========== 2.6. SuperAnnotate Detection ==========
    superannotate_score_val = 0.0
    try:
        from app.engine.superannotate_detector import superannotate_detector
        # Detect returns 0.0 (Human) to 1.0 (AI)
        superannotate_score_val = superannotate_detector.detect(text)
    except Exception as e:
        print(f"SuperAnnotate analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
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

    # ========== 4. Calculate Scores ==========
    
    # A. RoBERTa Score (Legacy - average of sentence scores)
    if sentence_scores_list:
        roberta_score = sum(s.ai_probability for s in sentence_scores_list) / len(sentence_scores_list)
    else:
        roberta_score = 0.0
    
    # B. Plagiarism Score (0-1) - Weighted by length
    plag_percent = (total_plagiarism_len / total_len) * 100
    plagiarism_score = min(1.0, plag_percent / 20)  # Strickland assumption: 20%+ = max score (stricter)
    # C. Statistical Metrics (Re-calculated for context)
    perplexity = ai_detector.calculate_perplexity(text[:2000])
    burstiness = ai_detector.analyze_burstiness(text)
    perplexity_flux = ai_detector.calculate_perplexity_flux(text[:3000])
    
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
    
    # D. Model-based score (RoBERTa)
    model_score = roberta_score
    ensemble_method = "roberta_stats"
    
    # E. Final AI Score: Use MAX of signals for aggressive detection
    # If any strong signal detected, use it as the score
    # This ensures high stylometry (1.0) or high model score results in high AI score
    
    # Calculate weighted average as baseline
    # Updated weights: 
    # - Model (40%): RoBERTa/Binoculars
    # - Statistical (25%): Burstiness/Perplexity
    # - Stylometry (35%): Phrase matching
    # - SuperAnnotate (Additive influence)
    weighted_score = (
        model_score * 0.40 +
        statistical_penalty * 0.25 +
        stylometry_score_val * 0.35 +
        superannotate_score_val * 0.40  # High weight for specialized model
    )
    weighted_score = min(1.0, weighted_score)
    
    # Use maximum of: weighted average, stylometry, model score
    # This ensures any strong signal dominates
    # Score Boosting: Map SuperAnnotate's conservative range (max ~0.75-0.80) to full 1.0 scale
    # If the specialized detector is > 0.7, it's almost certainly AI.
    boosted_sa_score = superannotate_score_val
    if superannotate_score_val > 0.7:
        boosted_sa_score = 0.9 + ((superannotate_score_val - 0.7) * 0.33)  # Map 0.7->0.9, 1.0->1.0
    elif superannotate_score_val > 0.5:
        boosted_sa_score = 0.6 + ((superannotate_score_val - 0.5) * 1.5)   # Map 0.5->0.6, 0.7->0.9

    ai_score = max(
        weighted_score, 
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
            boosted_score = (original_score * 0.4) + (ai_score * 0.6)
            
            # Update the sentence object
            sent.ai_probability = round(boosted_score, 4)
            sent.is_ai_generated = sent.ai_probability > 0.5
            
            # Recount flagged sentences
            ai_sentence_count = sum(1 for s in sentence_scores_list if s.is_ai_generated)
        
        # Sync segments with boosted scores
        if sentences and len(segments) == len(sentence_scores_list):
            for i, sent in enumerate(sentence_scores_list):
                 segments[i].ai_probability = sent.ai_probability

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
            roberta_score=round(roberta_score, 4),
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
