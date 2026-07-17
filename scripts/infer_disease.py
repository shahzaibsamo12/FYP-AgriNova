"""
AgriNova — Inference Script
Full pipeline: image → crop classifier → disease model → result

Usage:
    python infer_disease.py --image path/to/leaf.jpg
    python infer_disease.py --image path/to/leaf.jpg --crop rice     # skip classifier
    python infer_disease.py --batch path/to/folder/                  # batch inference
"""

import os
import sys
import cv2
import json
import pickle
import argparse
import warnings
import numpy as np
import tensorflow as tf
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_PATH  = os.path.join(PROJECT_ROOT, 'disease_detection', 'models')
IMAGE_SIZE   = (224, 224)


class AgriNovaInferenceEngine:
    """
    Production-grade inference engine for AgriNova disease detection.
    Mirrors the logic in disease_detection/services.py for standalone use.
    """

    BLUR_THRESHOLD = 100.0

    def __init__(self):
        self.crop_model    = None
        self.rice_model    = None
        self.cotton_model  = None
        self.crop_labels   = []
        self.rice_labels   = []
        self.cotton_labels = []
        self._load_all()

    def _load_model(self, name):
        path = os.path.join(MODELS_PATH, name)
        if os.path.exists(path):
            return tf.keras.models.load_model(path)
        return None

    def _load_labels(self, name):
        path = os.path.join(MODELS_PATH, name)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return []

    def _load_all(self):
        print('[INFO] Loading models...')

        self.crop_model    = self._load_model('crop_classifier.keras')
        self.crop_labels   = self._load_labels('crop_classifier_labels.pkl')

        self.rice_model    = self._load_model('rice_disease_model.keras')
        self.rice_labels   = self._load_labels('rice_model_labels.pkl')

        self.cotton_model  = self._load_model('cotton_disease_model.keras')
        self.cotton_labels = self._load_labels('cotton_model_labels.pkl')

        status = {
            'Crop Classifier': 'OK' if self.crop_model   else 'MISSING',
            'Rice Disease':    'OK' if self.rice_model    else 'MISSING (train first)',
            'Cotton Disease':  'OK' if self.cotton_model  else 'MISSING (train first)',
        }
        for k, v in status.items():
            print(f'  {k:<20} : {v}')

    def _check_blur(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _enhance(self, img_bgr):
        return cv2.detailEnhance(img_bgr, sigma_s=10, sigma_r=0.15)

    def _preprocess(self, img_bgr):
        """BGR → RGB → resize → [0,1] → batch dim."""
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMAGE_SIZE)
        img = img.astype('float32') / 255.0
        return np.expand_dims(img, axis=0)

    def predict(self, image_path, force_crop=None):
        """
        Parameters
        ----------
        image_path : str   Path to an image file.
        force_crop : str   'rice' or 'cotton' to skip the classifier (for testing).

        Returns
        -------
        dict with keys: crop, disease, confidence, quality_score, enhanced
        """
        if not os.path.exists(image_path):
            return {'error': f'File not found: {image_path}'}

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return {'error': 'Could not read image — check file format'}

        # Quality check
        quality = self._check_blur(img_bgr)
        enhanced = False
        if quality < self.BLUR_THRESHOLD:
            img_bgr  = self._enhance(img_bgr)
            enhanced = True

        inp = self._preprocess(img_bgr)

        # Step 1: Crop classification
        if force_crop:
            crop_type = force_crop.capitalize()
            crop_conf = 1.0
        else:
            if self.crop_model is None:
                return {'error': 'Crop classifier not loaded'}
            crop_probs = self.crop_model.predict(inp, verbose=0)[0]
            crop_idx   = int(np.argmax(crop_probs))
            raw_label  = self.crop_labels[crop_idx] if self.crop_labels else ''
            crop_type  = 'Cotton' if 'Cotton' in raw_label else 'Rice'
            crop_conf  = float(crop_probs[crop_idx])

        # Step 2: Disease detection
        if crop_type == 'Cotton':
            disease_model  = self.cotton_model
            disease_labels = self.cotton_labels
        else:
            disease_model  = self.rice_model
            disease_labels = self.rice_labels

        if disease_model is None:
            return {
                'crop':          crop_type,
                'crop_conf':     f'{crop_conf*100:.1f}%',
                'disease':       f'{crop_type} disease model not trained yet',
                'confidence':    None,
                'quality_score': round(quality, 2),
                'enhanced':      enhanced,
            }

        disease_probs = disease_model.predict(inp, verbose=0)[0]
        disease_idx   = int(np.argmax(disease_probs))
        disease_name  = disease_labels[disease_idx] if disease_labels else str(disease_idx)
        disease_conf  = float(disease_probs[disease_idx])

        # Top-3 predictions
        top3_idx  = np.argsort(disease_probs)[::-1][:3]
        top3 = [
            {
                'class':      disease_labels[i] if disease_labels else str(i),
                'confidence': f'{disease_probs[i]*100:.1f}%'
            }
            for i in top3_idx
        ]

        return {
            'crop':          crop_type,
            'crop_conf':     f'{crop_conf*100:.1f}%',
            'disease':       disease_name,
            'confidence':    f'{disease_conf*100:.1f}%',
            'top3':          top3,
            'quality_score': round(quality, 2),
            'enhanced':      enhanced,
        }


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='AgriNova Inference')
    parser.add_argument('--image',  default=None, help='Single image path')
    parser.add_argument('--batch',  default=None, help='Folder of images')
    parser.add_argument('--crop',   default=None, choices=['rice', 'cotton'],
                        help='Force crop type (skip classifier)')
    parser.add_argument('--pretty', action='store_true', help='Pretty-print JSON')
    args = parser.parse_args()

    engine = AgriNovaInferenceEngine()

    if args.image:
        result = engine.predict(args.image, force_crop=args.crop)
        indent = 2 if args.pretty else None
        print(json.dumps(result, indent=indent))

    elif args.batch:
        if not os.path.isdir(args.batch):
            print(f'[ERROR] Batch path is not a directory: {args.batch}')
            sys.exit(1)

        exts = ('.png', '.jpg', '.jpeg')
        images = [
            os.path.join(args.batch, f)
            for f in sorted(os.listdir(args.batch))
            if f.lower().endswith(exts)
        ]
        if not images:
            print('[ERROR] No images found in batch folder')
            sys.exit(1)

        print(f'[INFO] Processing {len(images)} images...\n')
        results = []
        for img_path in images:
            r = engine.predict(img_path, force_crop=args.crop)
            r['file'] = os.path.basename(img_path)
            results.append(r)
            status = f"  {r['file']:<40}  {r.get('crop','?'):<8}  {r.get('disease','?'):<40}  {r.get('confidence','?')}"
            print(status)

        # Save batch results
        import pandas as pd
        df = pd.DataFrame(results)
        out = os.path.join(args.batch, 'batch_results.csv')
        df.to_csv(out, index=False)
        print(f'\n[SAVED] {out}')

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
