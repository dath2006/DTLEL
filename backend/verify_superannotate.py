
import sys
import os
import requests
import json

# Ensure backend root is in path if running locally from root
sys.path.append(os.path.join(os.getcwd(), "backend"))

BASE_URL = "http://127.0.0.1:8000"

def test_superannotate_integration():
    print(f"Testing SuperAnnotate Integration against {BASE_URL}...")
    
    # 1. Test Human Text
    human_text = """
    The integration of renewable energy sources into the national grid presents both opportunities and challenges. 
    While solar and wind power offer sustainable alternatives to fossil fuels, their intermittent nature requires 
    robust energy storage solutions and grid modernization. Engineers are currently developing advanced battery 
    technologies to address these stability issues.
    """
    
    payload = {"text": human_text}
    try:
        response = requests.post(f"{BASE_URL}/api/v1/analyze/text", json=payload)
        response.raise_for_status()
        data = response.json()
        
        metrics = data.get("metrics", {})
        sa_score = metrics.get("superannotate_score")
        
        print("\n--- Human Text Result ---")
        print(f"AI Score: {data.get('ai_score')}")
        print(f"SuperAnnotate Score: {sa_score}")
        
        if sa_score is None:
            print("FAILED: superannotate_score missing from response")
            return
            
        if sa_score > 0.5:
             print("WARNING: Human text scored high on SuperAnnotate (False Positive?)")
        else:
             print("SUCCESS: Human text scored low as expected.")

    except Exception as e:
        print(f"Request failed: {e}")
        return

    # 2. Test AI Text (Synthetic Example)
    ai_text = """
    In the realm of digital transformation, artificial intelligence stands as a pivotal force, reshaping industries through automation and predictive analytics. By leveraging machine learning algorithms, organizations can unlock unprecedented insights from vast datasets, driving efficiency and innovation. This paradigm shift not only optimizes operational workflows but also creates new value propositions for customers in a rapidly evolving marketplace.
    """
    
    payload = {"text": ai_text}
    try:
        response = requests.post(f"{BASE_URL}/api/v1/analyze/text", json=payload)
        response.raise_for_status()
        data = response.json()
        
        metrics = data.get("metrics", {})
        sa_score = metrics.get("superannotate_score")
        
        print("\n--- AI Text Result ---")
        print(f"AI Score: {data.get('ai_score')}")
        print(f"SuperAnnotate Score: {sa_score}")
        
        if sa_score > 0.8:
             print("SUCCESS: AI text detected with high confidence.")
        else:
             print("WARNING: AI text score lower than expected.")

    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_superannotate_integration()
