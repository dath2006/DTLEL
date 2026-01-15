
import requests
import time

BASE_URL = "http://127.0.0.1:8000/api/v1"

def test_pipeline():
    print("=== Testing Plagiarism Pipeline v2 ===")
    
    # 1. Clear Data Pool
    print("\n1. Clearing Data Pool...")
    res = requests.delete(f"{BASE_URL}/documents/clear")
    print(res.json())
    
    # 2. Ingest Document
    print("\n2. Ingesting 'original.txt'...")
    original_text = "The quick brown fox jumps over the lazy dog. It was a broad, bright sunny day."
    files = {"file": ("original.txt", original_text)}
    res = requests.post(f"{BASE_URL}/ingest", files=files)
    print(res.json())
    
    # Allow indexing to sync (though it's sync in code, safe buffer)
    time.sleep(1)
    
    # 3. Analyze EXACT Copy (Should be > 0.95)
    print("\n3. Testing EXACT Copy...")
    req_body = {"text": "The quick brown fox jumps over the lazy dog."}
    res = requests.post(f"{BASE_URL}/analyze/text", json=req_body)
    data = res.json()
    if "plagiarism_score" not in data:
        print(f"   Error Response: {data}")
        return
    score = data["plagiarism_score"]
    print(f"   Score: {score} (Expected > 0.9)")
    if score > 0.9:
        print("   ✅ PASS: Exact copy detected.")
    else:
        print(f"   ❌ FAIL: Score too low ({score})")

    # 4. Analyze Paraphrase (Should be > 0.80)
    print("\n4. Testing AI Paraphrase...")
    # "The fast auburn fox leaps across the inactive canine."
    # Structure is similar, meaning is same. Cross-Encoder handles this.
    para_text = "The fast auburn fox leaps across the inactive canine."
    req_body = {"text": para_text}
    res = requests.post(f"{BASE_URL}/analyze/text", json=req_body)
    data = res.json()
    score = data["plagiarism_score"]
    # Inspect segments to see the match score
    best_segment_score = 0
    if data["segments"]:
        best_segment_score = max(s["plagiarism_score"] for s in data["segments"])
    
    print(f"   Plagiarism Score (Agg): {score}")
    print(f"   Best Segment Score (Re-Rank): {best_segment_score} (Expected > 0.75 for paraphrase)")
    
    if best_segment_score > 0.75:
        print("   ✅ PASS: Paraphrase detected.")
    else:
        print(f"   ❌ FAIL: Paraphrase missed ({best_segment_score})")

    # 5. Analyze Unrelated
    print("\n5. Testing Unrelated Text...")
    unrelated_text = "SpaceX launched a rocket into orbit yesterday."
    req_body = {"text": unrelated_text}
    res = requests.post(f"{BASE_URL}/analyze/text", json=req_body)
    data = res.json()
    score = data["plagiarism_score"]
    print(f"   Score: {score} (Expected < 0.2)")
    
    if score < 0.2:
        print("   ✅ PASS: Unrelated text clean.")
    else:
        print(f"   ❌ FAIL: False positive ({score})")

if __name__ == "__main__":
    try:
        test_pipeline()
    except Exception as e:
        print(f"Test failed: {e}")
