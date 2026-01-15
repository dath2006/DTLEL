
# Statistical Metrics Reliability Assessment

## 1. Burstiness (Is it reliable?)
**YES, it effectively distinguishes sentence structure variance.**

- **AI Text:** Consistently showed extremely low burstiness (~0.12). AI models tend to produce sentences of very similar average length.
- **Human Text:** Showed much higher variance (0.42 - 0.83).
    - *Human 1 (Creative):* 0.42 (Correctly identified as human-like)
    - *Human 3 (Dialogue):* 0.83 (Highly variable, correctly identified as human)

**Verdict:** Burstiness is a **strong** signal for catching "robotic" or monotonous writing styles common in AI.

## 2. Perplexity (Is GPT-2 reliable?)
**PARTIALLY. Use with caution.**

- **The Problem:** It flagged *everything* as low perplexity (Score ~15-20), even the human texts. 
    - *Human 2 (Wikipedia):* Scored 22.8 (Flagged as AI)
    - *Human 3 (Conversational):* Scored 20.3 (Flagged as AI)
- **Why?** GPT-2 is trained on WebText (which includes Wikipedia and Reddit). It finds almost *any* coherent English text "predictable."
- **Differentiation:** 
    - AI texts *did* score lower (13 - 18) compared to Humans (20 - 45), but the margin is very narrow.
    - It is **not** a reliable standalone detector.

## Recommendation
1. **Trust Burstiness:** Keep using it as a penalty. If `burstiness < 0.2`, it's almost certainly not natural creative human writing.
2. **Re-tune Perplexity Thresholds:** The current threshold (< 30 = AI) is too aggressive. It flags Wikipedia and formal writing as AI.
    - **Current:** `< 30` -> AI
    - **Proposed:** `< 12` -> AI (Very strict predictability only)
3. **Do NOT use Perplexity as a primary signal.** Use it only as a "tie-breaker" or weak signal. If a text is 99% AI classifier score AND has low perplexity, it confirms the result. But if Classifier says "Human" and Perplexity says "AI", trust the Classifier.
