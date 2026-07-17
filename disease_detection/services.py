"""
AgriNova — disease_detection/services.py  (FINAL PRODUCTION VERSION)

Drop-in replacement for the existing services.py.
Integrates trained rice_disease_model.keras and cotton_disease_model.keras
into the existing Flask blueprint without changing routes.py or app.py.

Changes vs existing services.py:
  1. Loads rice_disease_model.keras and cotton_disease_model.keras
     in addition to the crop classifier (already working).
  2. Adds model warm-up on startup to eliminate first-inference latency.
  3. Adds confidence gating — if confidence < threshold, returns a
     "low confidence" warning rather than a wrong prediction.
  4. Keeps all existing method signatures intact (predict, check_blur, etc.)
  5. Returns top-3 predictions in the JSON response.
"""

import os
import numpy as np
import cv2
import pickle
import tensorflow as tf
import torch
from config import Config
from core.utils import get_llm_response
from .models.crop_densenet import load_densenet


# ─────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────
IMAGE_SIZE          = (224, 224)
BLUR_THRESHOLD      = 100.0
MIN_CONFIDENCE      = 0.40   # below this → warn user


# ─────────────────────────────────────────────────────────────
#  Service class
# ─────────────────────────────────────────────────────────────
class DiseaseService:
    """
    Stateful service that holds all three models in memory.
    Instantiated once at module level → no per-request loading overhead.
    """

    def __init__(self):
        self.crop_model    = None
        self.rice_model    = None
        self.cotton_model  = None
        self.labels        = {}
        self._load_models()
        self._warm_up()

    # ── Model loading ─────────────────────────────────────────

    def _load_models(self):
        try:
            # Crop classifier (DenseNet PyTorch)
            try:
                self.crop_model         = load_densenet()
                self.labels['crop']     = ['Cotton', 'Rice']
                print('[OK] Crop classifier (DenseNet PyTorch) loaded')
            except Exception as e:
                print(f'[WARN] Failed to load crop classifier DenseNet PyTorch: {e}')

            # Cotton disease model (Stage 2)
            if os.path.exists(Config.COTTON_MODEL_PATH):
                self.cotton_model       = tf.keras.models.load_model(Config.COTTON_MODEL_PATH)
                self.labels['cotton']   = self._load_labels(Config.COTTON_MODEL_PATH)
                print('[OK] Cotton disease model loaded')
            else:
                print('[WARN] cotton_disease_model.keras not found — train Stage 2 first')

            # Rice disease model (Stage 2)
            if os.path.exists(Config.RICE_MODEL_PATH):
                self.rice_model         = tf.keras.models.load_model(Config.RICE_MODEL_PATH)
                self.labels['rice']     = self._load_labels(Config.RICE_MODEL_PATH)
                print('[OK] Rice disease model loaded')
            else:
                print('[WARN] rice_disease_model.keras not found — train Stage 2 first')

        except Exception as e:
            print(f'[ERROR] Model loading failed: {e}')

    def _load_labels(self, model_path):
        """Derive label path from model path and load pickle."""
        label_path = model_path.replace('.keras', '_labels.pkl')
        if not os.path.exists(label_path):
            label_path = model_path.replace('_disease_model.keras', '_model_labels.pkl')
        if os.path.exists(label_path):
            with open(label_path, 'rb') as f:
                return pickle.load(f)
        return []

    def _warm_up(self):
        """Run a dummy forward pass on each loaded model to JIT-compile graph."""
        # 1. Warm up Keras models (rice, cotton)
        dummy_keras = np.zeros((1, *IMAGE_SIZE, 3), dtype='float32')
        for name, model in [('rice', self.rice_model),
                             ('cotton', self.cotton_model)]:
            if model is not None:
                try:
                    model.predict(dummy_keras, verbose=0)
                    print(f'[OK] {name} model warmed up')
                except Exception as e:
                    print(f'[WARN] Warm-up failed for {name}: {e}')

        # 2. Warm up PyTorch model (crop classifier)
        if self.crop_model is not None:
            try:
                dummy_torch = torch.zeros((1, 3, *IMAGE_SIZE), dtype=torch.float32)
                with torch.no_grad():
                    self.crop_model(dummy_torch)
                print('[OK] crop (DenseNet PyTorch) model warmed up')
            except Exception as e:
                print(f'[WARN] Warm-up failed for crop (DenseNet PyTorch): {e}')

    # ── Image utilities ───────────────────────────────────────

    def check_blur(self, img_path):
        """Returns (is_blurry: bool, laplacian_variance: float)."""
        img = cv2.imread(img_path)
        if img is None:
            return False, 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        var  = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(var) < BLUR_THRESHOLD, float(var)

    def upscale_image(self, img_path):
        """Apply detail enhancement on blurry images."""
        img = cv2.imread(img_path)
        if img is None:
            return img_path
        enhanced = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
        base, ext = os.path.splitext(img_path)
        enhanced_path = f'{base}_enhanced{ext}'
        cv2.imwrite(enhanced_path, enhanced)
        return enhanced_path

    def preprocess_image(self, img_path):
        """
        Load → RGB → resize → [0,1] → add batch dimension.
        Consistent with training pipeline.
        """
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f'Cannot read image: {img_path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMAGE_SIZE)
        img = img.astype('float32') / 255.0
        return np.expand_dims(img, axis=0)

    def preprocess_crop_image(self, img_path):
        """
        Load → RGB → resize → [0,1] → Normalize with ImageNet mean/std → (C, H, W) → add batch dimension.
        Expected format for PyTorch DenseNet.
        """
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f'Cannot read image: {img_path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMAGE_SIZE)
        img = img.astype('float32') / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype='float32')
        std = np.array([0.229, 0.224, 0.225], dtype='float32')
        img = (img - mean) / std
        
        img = img.transpose(2, 0, 1)
        return np.expand_dims(img, axis=0)

    # ── Core prediction ───────────────────────────────────────

    def predict(self, img_path, language='en'):
        """
        Main prediction pipeline.

        Returns dict:
            crop, disease, confidence, advice, top3, note, warning (optional)
        """
        if self.crop_model is None:
            return {'error': 'Crop classifier not loaded. Check model paths in config.py'}

        enhanced_path = None

        try:
            # Step 1 — Quality check & enhancement
            is_blurry, quality_score = self.check_blur(img_path)
            processing_note = ''
            if is_blurry:
                enhanced_path = self.upscale_image(img_path)
                img_path      = enhanced_path
                processing_note = f'Image quality enhanced (blur score: {quality_score:.1f})'

            # Step 2 — Crop classification (using PyTorch DenseNet)
            crop_inp   = self.preprocess_crop_image(img_path)
            crop_tensor = torch.from_numpy(crop_inp)
            with torch.no_grad():
                crop_logits = self.crop_model(crop_tensor)
                crop_probs = torch.softmax(crop_logits, dim=1).numpy()[0]
            crop_idx   = int(np.argmax(crop_probs))
            raw_label  = self.labels.get('crop', ['Cotton', 'Rice'])[crop_idx]
            crop_type  = 'Cotton' if 'Cotton' in raw_label else 'Rice'
            crop_conf  = float(crop_probs[crop_idx])

            # Step 3 — Disease detection
            if crop_type == 'Cotton':
                disease_model  = self.cotton_model
                disease_labels = self.labels.get('cotton', [])
            else:
                disease_model  = self.rice_model
                disease_labels = self.labels.get('rice', [])

            if disease_model is None:
                return {
                    'crop':       crop_type,
                    'crop_conf':  f'{crop_conf*100:.1f}%',
                    'disease':    'Disease model not yet trained',
                    'confidence': None,
                    'advice':     (
                        f'The crop classifier identified this as a {crop_type} plant. '
                        f'Train the {crop_type} disease model to get a full diagnosis.'
                    ),
                    'note':       processing_note,
                }

            inp = self.preprocess_image(img_path)
            disease_probs = disease_model.predict(inp, verbose=0)[0]
            disease_idx   = int(np.argmax(disease_probs))
            disease_conf  = float(disease_probs[disease_idx])
            disease_name  = (disease_labels[disease_idx]
                             if disease_labels else str(disease_idx))

            # Top-3 predictions
            top3_idx = np.argsort(disease_probs)[::-1][:3]
            top3 = [
                {
                    'class':      disease_labels[i] if disease_labels else str(i),
                    'confidence': f'{disease_probs[i]*100:.1f}%'
                }
                for i in top3_idx
            ]

            # Step 4 — Confidence gate
            warning = ''
            if disease_conf < MIN_CONFIDENCE:
                warning = (
                    f'Low confidence ({disease_conf*100:.1f}%). '
                    'Please upload a clearer, closer image of the leaf.'
                )

            # Step 5 — AI advice
            advice = self._get_ai_advice(crop_type, disease_name, language)

            result = {
                'crop':       crop_type,
                'disease':    disease_name,
                'confidence': f'{disease_conf*100:.2f}%',
                'top3':       top3,
                'advice':     advice,
                'note':       processing_note,
            }
            if warning:
                result['warning'] = warning

            return result

        finally:
            # Clean up enhanced image if created
            if enhanced_path and os.path.exists(enhanced_path):
                os.remove(enhanced_path)

    # ── AI advice ─────────────────────────────────────────────

    def _get_ai_advice(self, crop, disease, language):
        lang_str = 'English' if language == 'en' else 'Urdu script (Urdu language)'
        prompt   = (
            f"Expert agricultural advice for {crop} plant with '{disease}'.\n"
            f"Respond in {lang_str}.\n\n"
            f"Structure your response exactly as follows (using Markdown):\n"
            f"### 1. Disease Description\n"
            f"[Provide a brief disease description in 2-3 sentences]\n\n"
            f"### 2. Immediate Treatment Steps\n"
            f"[Provide a single-level bulleted list. Do NOT use nested bullet points. Each item should start with a single bullet '*' followed by a bold title, a colon, and the description on the same line, e.g., '* **Action Item:** Description of action.']\n\n"
            f"### 3. Prevention Tips\n"
            f"[Provide a single-level bulleted list. Do NOT use nested bullet points. Each item should start with a single bullet '*' followed by a bold title, a colon, and the description on the same line, e.g., '* **Tip:** Description of tip.']\n\n"
            f"Constraints:\n"
            f"- Do NOT nest bullets. Keep lists completely flat (single level).\n"
            f"- Use simple, farmer-friendly language with local product names where applicable."
        )
        return get_llm_response(prompt)


# ─────────────────────────────────────────────────────────────
#  Singleton — imported by routes.py
# ─────────────────────────────────────────────────────────────
disease_service = DiseaseService()
