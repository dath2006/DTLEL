
import asyncio
from app.engine.detector import ai_detector

# Test Data
human_texts = [
    # 1. Casual / Creative Writing
    """The old house groaned under the weight of the storm. It was as if the wood itself could feel the pressure of the wind, aching to give way but holding on for one more night. I sat by the window, watching the rain blur the world outside into a watercolor painting of greys and blues. My coffee had gone cold hours ago, but I couldn't bring myself to move.""",
    
    # 2. Technical / Formal Writing (Wikipedia)
    """Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured (particularly procedural), object-oriented and functional programming.""",
    
    # 3. Burstiness Example (Varied sentence lengths)
    """No. I don't think so. That's a terrible idea, honestly, because it ignores the fundamental principles of thermodynamics that we've spent centuries validating through rigorous experimentation. It just won't work."""
]

ai_texts = [
    # 1. GPT-3.5 Style Explanation
    """The phenomenon of bioluminescence is a fascinating adaptation seen in various marine organisms. It serves multiple purposes, including attracting mates, deterring predators, and luring prey. The chemical reaction that produces this light involves a molecule called luciferin and an enzyme called luciferase. This process is highly efficient, producing very little heat compared to traditional light sources.""",
    
    # 2. GPT-4 Style Formal
    """In recent years, the field of artificial intelligence has witnessed significant advancements, particularly in the domain of natural language processing. Large language models (LLMs) have demonstrated an unprecedented ability to generate human-like text, answer complex queries, and assist in creative tasks. However, these developments also raise important ethical considerations regarding data privacy and the potential for misuse.""",
    
    # 3. Low Burstiness Example (Uniform sentence lengths)
    """The weather today is very sunny and warm. It is a perfect day to go to the park. Many people are walking their dogs outside. The birds are singing in the trees nearby. Everyone seems to be enjoying the fresh air."""
]

def analyze_sample(label: str, text: str):
    print(f"\n--- {label} ---")
    print(f"Snippet: {text[:60]}...")
    
    # 1. Perplexity (GPT-2)
    ppl = ai_detector.calculate_perplexity(text)
    
    # 2. Burstiness (Std Dev / Mean of sentence lengths)
    burst = ai_detector.analyze_burstiness(text)
    
    print(f"Perplexity (Predictability): {ppl:.2f}")
    print(f"Burstiness (Variation):      {burst:.2f}")
    
    # Interpretation based on common thresholds
    # Perplexity: < 30 usually AI, > 60 usually Human
    # Burstiness: < 0.4 usually AI, > 0.6 usually Human
    
    ai_signals = []
    if ppl < 30: ai_signals.append("Low Perplexity (AI-like)")
    if burst < 0.4: ai_signals.append("Low Burstiness (AI-like)")
    
    if not ai_signals:
        print(">> Verdict: Likely HUMAN")
    else:
        print(f">> Verdict: Suspicious ({', '.join(ai_signals)})")

async def run_test():
    print("Initializing Statistical Metrics Test...")
    print("------------------------------------------")
    
    print("\n[ HUMAN TEXT SAMPLES ]")
    for i, text in enumerate(human_texts):
        analyze_sample(f"Human {i+1}", text)
        
    print("\n[ AI GENERATED SAMPLES ]")
    for i, text in enumerate(ai_texts):
        analyze_sample(f"AI {i+1}", text)

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_test())
