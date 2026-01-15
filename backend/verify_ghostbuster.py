import requests
import json

BASE_URL = "http://127.0.0.1:8000/api/v1"

def test_ghostbuster_features():
    print("Testing /analyze/text for Ghostbuster features...")
    payload = {
        "text": "Artificial intelligence creates content with uniform probability distributions."
    }
    
    try:
        r = requests.post(f"{BASE_URL}/analyze/text", json=payload)
        
        if r.status_code != 200:
            print(f"❌ Request failed with status {r.status_code}")
            print(r.text)
            return
            
        data = r.json()
        
        # Check if ghostbuster field exists
        if "ghostbuster" in data and data["ghostbuster"]:
            print("✅ Ghostbuster features present:")
            gb = data["ghostbuster"]
            print(f"  - Score: {gb.get('ghostbuster_score')}")
            print(f"  - Mean Unigram Prob: {gb.get('unigram_mean')}")
            print(f"  - Prob Variance: {gb.get('prob_variance')}")
        else:
            print("❌ Ghostbuster features output missing from response")
            print("Keys found:", list(data.keys()))
            
        # Check if score is integrated into metrics
        if "metrics" in data:
            print(f"✅ Metrics Ghostbuster Score: {data['metrics'].get('ghostbuster_score')}")
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")

if __name__ == "__main__":
    test_ghostbuster_features()
