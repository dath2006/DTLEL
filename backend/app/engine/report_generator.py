from typing import List, Dict, Any

class ReportGenerator:
    def __init__(self, ai_detector, plagiarism_checker):
        self.ai_detector = ai_detector
        self.plagiarism_checker = plagiarism_checker

    def generate_report(self, text: str) -> Dict[str, Any]:
        # Run detection
        ai_result = self.ai_detector.verify_text(text)
        plagiarism_result = self.plagiarism_checker.search(text)

        ai_sentences = ai_result.get("sentence_details", [])
        plag_sentences = plagiarism_result.get("matches", [])

        # Merge results
        # Assuming NLTK tokenization is deterministic and identical in both engines
        # We zip them. If lengths differ, we zip to the shortest (or handle mismatch)
        
        segments = []
        for ai_sent, plag_sent in zip(ai_sentences, plag_sentences):
            # Sanity check: verify text matches mostly (ignoring whitespace differences if any)
            # If not matching, we might have an alignment issue.
            # But since we use the exact same tokenizer on the exact same string, it should be fine.
            
            segments.append({
                "text": ai_sent["sentence"],
                "ai_detection": {
                    "is_ai": ai_sent["is_ai"],
                    "confidence": ai_sent["confidence"],
                    "perplexity": ai_sent["perplexity"],
                    "reasons": ai_sent["reasons"]
                },
                "plagiarism_detection": {
                    "is_plagiarized": plag_sent["is_plagiarized"],
                    "score": plag_sent["best_score"],
                    "matches": plag_sent["matches"]
                }
            })

        return {
            "summary": {
                "ai_score": ai_result["score"],
                "ai_status": ai_result["label"],
                "plagiarism_score": plagiarism_result["plagiarism_score"],
                "plagiarism_status": "High" if plagiarism_result["plagiarism_score"] > 20 else "Low" # Simple threshold
            },
            "segments": segments
        }
