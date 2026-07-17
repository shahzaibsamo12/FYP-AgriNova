import os
import numpy as np
import tensorflow as tf
import cv2
import pickle
from utils_gradcam import make_gradcam_heatmap, save_and_display_gradcam

# --- Configuration ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "crop_classifier.keras")
LABEL_PATH = os.path.join(os.path.dirname(__file__), "models", "crop_classifier_labels.pkl")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "diagnostics")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_diagnostic(img_path):
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}. Please run training first!")
        return

    # 1. Load Model and Labels
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABEL_PATH, "rb") as f:
        labels = pickle.load(f)

    # 2. Preprocess Image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0)

    # 3. Generate Prediction
    preds = model.predict(img_array)
    idx = np.argmax(preds[0])
    class_name = labels[idx]
    confidence = preds[0][idx]

    print(f"Prediction: {class_name} ({confidence:.2f})")

    # 4. Generate Grad-CAM
    # For EfficientNetV2B0, the last conv layer is 'top_conv'
    heatmap = make_gradcam_heatmap(img_array, model, "top_conv")

    # 5. Save Visualization
    output_path = os.path.join(OUTPUT_DIR, f"targeting_{os.path.basename(img_path)}")
    save_and_display_gradcam(img_path, heatmap, cam_path=output_path)
    
    print(f"[SUCCESS] Diagnostic saved to: {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python test_targeting.py <path_to_image>")
    else:
        run_diagnostic(sys.argv[1])
