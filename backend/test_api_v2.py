import requests
import time
import json

URL = "http://localhost:8000/api/v1/analyze/text"
PAYLOAD = {"text": "This is a test of the optimized backend. It should be processed using ONNX runtime if enabled."}

def test_api():
    print(f"Testing API: {URL}")
    
    # Warmup
    try:
        requests.post(URL, json=PAYLOAD)
    except Exception as e:
        print(f"Error connecting to API: {e}")
        return

    start = time.time()
    response = requests.post(URL, json=PAYLOAD)
    end = time.time()
    
    latency = (end - start) * 1000
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Latency: {latency:.2f} ms")
        print(f"AI Score: {data.get('ai_score')}")
        print("Success!")
    else:
        print(f"Failed: {response.text}")

if __name__ == "__main__":
    test_api()
