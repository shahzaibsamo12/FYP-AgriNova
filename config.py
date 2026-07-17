"""
AgriNova — config.py  (UPDATED for Stage 2)

Only the Disease Detection model paths are updated.
Everything else is unchanged from the original.
"""

import os


class Config:
    # API Keys
    MISTRAL_API_KEY  = os.getenv("MISTRAL_API_KEY", "5qwy7IvN5e3NUNqWdDpMN0ylKSSFkdMA")
    MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"

    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Fertilizer Recommendation (unchanged)
    FERTILIZER_MODEL_PATH = os.path.join(
        BASE_DIR, "fertilizer_recommendation", "models", "fertilizer_model_bundle.pkl"
    )

    # ── Disease Detection (Stage 1 + Stage 2) ──────────────────
    CROP_MODEL_PATH   = os.path.join(
        BASE_DIR, "disease_detection", "models", "densenet_cotton_rice_classifier.pth"
    )
    # Stage 2 models — train with train_disease.py to generate these
    RICE_MODEL_PATH   = os.path.join(
        BASE_DIR, "disease_detection", "models", "rice_disease_model.keras"
    )
    COTTON_MODEL_PATH = os.path.join(
        BASE_DIR, "disease_detection", "models", "cotton_disease_model.keras"
    )

    # Image Uploads
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

    # App Settings
    SECRET_KEY     = os.getenv("SECRET_KEY", "dev-secret-key")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB upload limit
