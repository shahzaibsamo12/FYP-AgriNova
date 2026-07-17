import pickle
import numpy as np
import os
import time
from config import Config
from core.utils import get_llm_response

class FertilizerService:
    def __init__(self):
        self.fertilizer_bundle = None
        self.load_model()
        self.warm_up()

    def load_model(self):
        try:
            if os.path.exists(Config.FERTILIZER_MODEL_PATH):
                with open(Config.FERTILIZER_MODEL_PATH, "rb") as f:
                    self.fertilizer_bundle = pickle.load(f)
                print(f"[OK] Loaded Fertilizer Model from {Config.FERTILIZER_MODEL_PATH}")
            else:
                print(f"[WARNING] Fertilizer Model not found at {Config.FERTILIZER_MODEL_PATH}")
        except Exception as e:
            print(f"[ERROR] Error loading fertilizer model: {e}")

    def warm_up(self):
        """
        Runs dummy predictions to 'wake up' libraries and initialize the AI API connection.
        """
        if not self.fertilizer_bundle:
            return
            
        print("[INFO] Warming up Fertilizer module (ML & AI)...")
        try:
            # 1. Warm up ML Model (fast)
            dummy_data = {
                "Nitrogen": 50, "Phosphorous": 50, "Potassium": 50,
                "Temperature": 25, "Humidity": 50, "Moisture": 20,
                "Soil Type": "Loamy Soil", "Crop Type": "Rice",
                "language": "en"
            }
            self.predict(dummy_data, skip_llm=True)
            
            # 2. Warm up AI Connection (prevents first-request timeout)
            # We use a very trivial prompt to minimize cost/time
            get_llm_response("Connection warm-up. Reply with 'OK'.")
            
            print("[OK] Fertilizer module warm-up complete.")
        except Exception as e:
            print(f"[WARNING] Warm-up failed: {e}")


    def get_config(self):
        if not self.fertilizer_bundle:
            return {"error": "Model not loaded"}
        
        return {
            "features": self.fertilizer_bundle["features"],
            "num_cols": self.fertilizer_bundle["num_cols"],
            "cat_cols": self.fertilizer_bundle["cat_cols"],
            "options": {
                "Crop Type": ["Rice", "Wheat", "Cotton"],
                "Soil Type": ["Loamy Soil", "Peaty Soil", "Neutral Soil", "Acidic Soil"]
            }
        }

    def predict(self, data, skip_llm=False):
        if not self.fertilizer_bundle:
            raise Exception("Model not loaded")

        t_start = time.time()
        print(f"[DEBUG] Starting prediction for input: {data.get('Crop Type')}...")

        # Extract components
        model = self.fertilizer_bundle["fert_model"]
        fert_encoder = self.fertilizer_bundle["fert_encoder"]
        scaler = self.fertilizer_bundle["scaler"]
        feature_encoders = self.fertilizer_bundle["feature_encoders"]
        features = self.fertilizer_bundle["features"]
        num_cols = self.fertilizer_bundle["num_cols"]
        cat_cols = self.fertilizer_bundle["cat_cols"]

        # Prepare Input
        input_data = []
        for feature in features:
            if feature in cat_cols:
                val = data.get(feature)
                if val not in feature_encoders[feature].classes_:
                    val = feature_encoders[feature].classes_[0] # Fallback
                val = feature_encoders[feature].transform([val])[0]
            else:
                val = float(data.get(feature, 0))
            input_data.append(val)

        X_user = np.array(input_data).reshape(1, -1)

        # Scale
        X_num = X_user[:, [features.index(f) for f in num_cols if f in features]]
        X_cat = X_user[:, [features.index(f) for f in cat_cols if f in features]]
        X_num_scaled = scaler.transform(X_num)
        X_user_final = np.concatenate([X_num_scaled, X_cat], axis=1)

        # Predict
        pred_idx = model.predict(X_user_final)[0]
        fertilizer_pred = fert_encoder.inverse_transform([pred_idx])[0]
        
        t_ml = time.time()
        print(f"[DEBUG] ML Prediction took {t_ml - t_start:.4f}s. Result: {fertilizer_pred}")

        if skip_llm:
            return {"fertilizer": fertilizer_pred}

        # Get AI Advice
        import json
        lang_prompt = "Provide the response in English." if data.get('language') == 'en' else "Provide the response in Urdu (Prop Urdu script, NOT Devanagari/Hindi)."
        
        advice_prompt = (
            f"You are an expert agronomist. Provide the translated/transliterated name, a simple remark, and expert advice for the fertilizer '{fertilizer_pred}'.\n"
            f"{lang_prompt}\n\n"
            f"You MUST return the output as a valid JSON object. Do not include any extra text or explanations. "
            f"The response must exactly start with '{{' and end with '}}'.\n\n"
            f"JSON schema:\n"
            f"{{\n"
            f"  \"name\": \"[Name of the fertilizer ONLY in the requested language. If Urdu, provide it in Urdu script. If English, provide in English. NO extra synonyms.]\",\n"
            f"  \"remark\": \"[Write a single, simple, precise sentence about what this fertilizer does]\",\n"
            f"  \"advice\": \"## Key Benefits\\n- [Benefit point 1]\\n- [Benefit point 2]\\n\\n## Usage Instructions\\n- [Instruction point 1]\\n- [Instruction point 2]\\n\\n## Safety Precautions\\n- [Precaution point 1]\"\n"
            f"}}\n"
        )
        
        t_llm_start = time.time()
        llm_response = get_llm_response(advice_prompt)
        t_llm_end = time.time()
        print(f"[DEBUG] AI Advice API call took {t_llm_end - t_llm_start:.4f}s.")
        
        # Parse Response
        final_fertilizer = fertilizer_pred # Default to English
        final_remark = "No remarks available." 
        final_advice = llm_response

        try:
            # Clean up potential markdown JSON wrappers if generated by the LLM
            cleaned_response = llm_response.strip()
            if cleaned_response.startswith("```"):
                lines = cleaned_response.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                cleaned_response = "\n".join(lines).strip()
            
            parsed = json.loads(cleaned_response)
            final_fertilizer = parsed.get("name", fertilizer_pred).strip()
            final_remark = parsed.get("remark", "No remarks available.").strip()
            final_advice = parsed.get("advice", llm_response).strip()
        except Exception as e:
            print(f"[WARNING] Could not parse LLM JSON response: {e}. Falling back to text splitting.")
            try:
                if "X_NAME_START" in llm_response:
                    parts = llm_response.split("X_NAME_START")[1].split("X_NAME_END")
                    if len(parts) > 1:
                        final_fertilizer = parts[0].strip()

                if "X_REMARK_START" in llm_response:
                    parts = llm_response.split("X_REMARK_START")[1].split("X_REMARK_END")
                    if len(parts) > 1:
                        final_remark = parts[0].strip()
                
                if "X_ADVICE_START" in llm_response:
                    parts = llm_response.split("X_ADVICE_START")[1].split("X_ADVICE_END")
                    if len(parts) > 0:
                        final_advice = parts[0].strip().replace('\r', '')
            except Exception as fe:
                print(f"[WARNING] Fallback text splitting also failed: {fe}")

        total_time = time.time() - t_start
        print(f"[DEBUG] Total request handling time: {total_time:.4f}s.")

        return {
            "fertilizer": final_fertilizer,
            "remarks": final_remark,
            "advice": final_advice
        }

fertilizer_service = FertilizerService()

