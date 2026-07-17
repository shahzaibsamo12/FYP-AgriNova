import os
import cv2
import albumentations as A
import numpy as np
from tqdm import tqdm

# --- Configuration ---
BASE_PATH = os.path.join(os.path.dirname(__file__), "Rice_Cotton_disease_Data")

# Specific Guaranteed Pipelines
pipelines = {
    "flip": A.Compose([A.HorizontalFlip(p=1.0)]),
    "rotate": A.Compose([A.Rotate(limit=30, p=1.0)]),
    "blur": A.Compose([A.GaussianBlur(blur_limit=(3, 7), p=1.0)]),
    "brightness": A.Compose([A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0)])
}

def augment_folder(original_dir, target_dir):
    if not os.path.exists(original_dir):
        print(f"[SKIP] {original_dir} not found.")
        return

    os.makedirs(target_dir, exist_ok=True)
    
    for root, dirs, files in os.walk(original_dir):
        rel_path = os.path.relpath(root, original_dir)
        target_subdir = os.path.abspath(os.path.join(target_dir, rel_path))
        os.makedirs(target_subdir, exist_ok=True)
        
        print(f"Processing Folder: {rel_path if rel_path != '.' else 'Root'}")
        
        for file in tqdm(files):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(root, file)
            image = cv2.imread(img_path)
            if image is None: continue
            
            # Convert to RGB for Albumentations
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            base_name = os.path.splitext(file)[0]
            ext = os.path.splitext(file)[1]
            
            # Apply EACH guaranteed transform
            for key, pipeline in pipelines.items():
                augmented = pipeline(image=image)['image']
                # Convert back to BGR for saving
                aug_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                
                new_name = f"{base_name}_{key}{ext}"
                cv2.imwrite(os.path.join(target_subdir, new_name), aug_bgr)

if __name__ == "__main__":
    print("--- Starting Deterministic Dataset Expansion (Guaranteed Transforms) ---")
    
    # 1. Cotton Data
    augment_folder(
        os.path.join(BASE_PATH, "Cotton_Disease_Data", "Cotton"),
        os.path.join(os.path.dirname(__file__), "AGRINOVA_DATASET", "Cotton_Augmented")
    )
    
    # 2. Rice Data
    augment_folder(
        os.path.join(BASE_PATH, "Rice_Disease_Data", "Rice"),
        os.path.join(os.path.dirname(__file__), "AGRINOVA_DATASET", "Rice_Augmented")
    )
    
    print("\n--- [SUCCESS] Guaranteed Dataset Expanded on Disk ---")
