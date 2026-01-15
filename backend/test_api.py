import requests
import time

BASE_URL = "http://127.0.0.1:8000/api/v1"

def test_health():
    print("Testing /health...")
    try:
        r = requests.get("http://127.0.0.1:8000/health")
        print(r.json())
        assert r.status_code == 200
        print("✅ Health Check Passed")
    except Exception as e:
        print(f"❌ Health Check Failed: {e}")

def test_ingest():
    print("\nTesting /ingest...")
    files = {'file': ('source.txt', 'This is a source text about artificial intelligence and its implications on society.', 'text/plain')}
    try:
        r = requests.post(f"{BASE_URL}/ingest", files=files)
        print(r.json())
        assert r.status_code == 200
        print("✅ Ingest Passed")
    except Exception as e:
        print(f"❌ Ingest Failed: {e}")

def test_analyze():
    print("\nTesting /analyze...")
    # Test with similar text (Plagiarism)
    files = {'file': ('query.txt', 'This is a source text about artificial intelligence and its implications.', 'text/plain')}
    try:
        r = requests.post(f"{BASE_URL}/analyze", files=files)
        data = r.json()
        print(f"Integrity Score: {data['overall_integrity_score']}")
        print(f"AI Prob: {data['metrics']['ai_score']}")
        print(f"Plagiarism %: {data['metrics']['plagiarism_percentage']}")
        assert r.status_code == 200
        print("✅ Analyze Passed")
    except Exception as e:
        print(f"❌ Analyze Failed: {e}")

if __name__ == "__main__":
    # Wait for server to start if running via script
    time.sleep(2) 
    test_health()
    test_ingest()
    test_analyze()
