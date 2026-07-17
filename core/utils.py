import requests
from config import Config

def get_llm_response(prompt, system_role="You are a helpful assistant."):
    """
    Generic function to call LLM (Mistral).
    """
    headers = {
        "Authorization": f"Bearer {Config.MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "mistral-large-latest",
        "messages": [
            {"role": "system", "content": system_role},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        print(f"[DEBUG] Calling LLM API with prompt length: {len(prompt)}")
        response = requests.post(Config.MISTRAL_ENDPOINT, headers=headers, json=payload, timeout=60) # 60s timeout
        response.raise_for_status()
        print("[DEBUG] LLM Response received successfully")
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        print("[ERROR] LLM Request timed out")
        return "I'm sorry, the AI service is taking too long to respond. Please try again."
    except Exception as e:
        print(f"[ERROR] LLM Error: {e}")
        return "I'm sorry, I'm having trouble connecting to my knowledge base right now. Please try again later."
