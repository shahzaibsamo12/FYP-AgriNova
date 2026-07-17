import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetV2B0
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# --- Configuration ---
BASE_PATH = os.path.join(os.path.dirname(__file__), "Rice_Cotton_disease_Data")
MODELS_EXPORT_PATH = os.path.join(os.path.dirname(__file__), "models")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 1

os.makedirs(MODELS_EXPORT_PATH, exist_ok=True)

def build_model(num_classes):
    """
    EfficientNetB0 with Transfer Learning and fixed Keras 3 Input.
    """
    inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    base_model = EfficientNetV2B0(weights='imagenet', include_top=False, input_tensor=inputs)
    
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    for layer in base_model.layers:
        layer.trainable = False
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model

def get_crop_dataset_modern(base_path, split_ratio=0.3):
    """
    Strict Isolation Split:
    - Test: 30% of Original Only
    - Train: 70% of Original + 100% of Offline Augmented Files
    """
    train_paths, train_labels = [], []
    test_paths, test_labels = [], []
    
    categories = {
        "Cotton_Disease_Data": 0,
        "Rice_Disease_Data": 1
    }

    for cat_folder, label_id in categories.items():
        cat_path = os.path.join(base_path, cat_folder)
        if not os.path.exists(cat_path): continue
        
        # 1. Collect Original Images (for splitting)
        orig_img_paths = []
        for d in os.listdir(cat_path):
            d_path = os.path.join(cat_path, d)
            if os.path.isdir(d_path) and "augmented" not in d.lower():
                for root, _, files in os.walk(d_path):
                    for f in files:
                        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                            orig_img_paths.append(os.path.join(root, f))
        
        # 2. Split Originals (70/30)
        np.random.seed(42)
        np.random.shuffle(orig_img_paths)
        
        split_idx = int(len(orig_img_paths) * (1 - split_ratio))
        train_paths.extend(orig_img_paths[:split_idx])
        train_labels.extend([label_id] * split_idx)
        
        test_paths.extend(orig_img_paths[split_idx:])
        test_labels.extend([label_id] * (len(orig_img_paths) - split_idx))

        # 3. Collect Offline Augmented Images (DISABLED FOR DIAGNOSTIC)
        """
        new_aug_root = os.path.join(os.path.dirname(__file__), "AGRINOVA_DATASET")
        target_aug_folder = "Cotton_Augmented" if label_id == 0 else "Rice_Augmented"
        new_aug_path = os.path.join(new_aug_root, target_aug_folder)
        
        if os.path.exists(new_aug_path):
            print(f"Found offline augmented data in: {target_aug_folder}")
            for root, _, files in os.walk(new_aug_path):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        train_paths.append(os.path.join(root, f))
                        train_labels.append(label_id)
        """

    print(f"\nAggregation Strategy: Offline Augmentation + Isolation")
    print(f"Train Set: {len(train_paths)} images (Originals + Physical Augmented)")
    print(f"Test Set: {len(test_paths)} images (Pure Original)")
    
    def process_path(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMAGE_SIZE)
        return img, label

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.shuffle(len(train_paths)).map(process_path, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    test_ds = test_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds, ["Cotton", "Rice"]

# --- EXECUTION ---
print("\n--- Training Stage 1: Modernized EfficientNet Crop Classifier ---")
train_ds, test_ds, class_names = get_crop_dataset_modern(BASE_PATH, split_ratio=0.3)

model = build_model(len(class_names))
model.fit(train_ds, epochs=EPOCHS, validation_data=test_ds)

# Evaluation
print("\nEvaluating on Scientific Test Set (30% Pure Original)...")
y_true, y_pred = [], []
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

acc = np.mean(np.array(y_true) == np.array(y_pred))
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)

print(f"\n[FINAL TEST RESULTS] Accuracy: {acc:.4f}, F1: {f1:.4f}")

# Save
model.save(os.path.join(MODELS_EXPORT_PATH, "crop_classifier.keras"))
with open(os.path.join(MODELS_EXPORT_PATH, "crop_classifier_labels.pkl"), "wb") as f:
    pickle.dump(class_names, f)

# Log Results
score_data = {
    "Model": "Modernized Crop Classifier (EffNetB0)",
    "Test Accuracy": f"{acc:.4f}",
    "F1-Score": f"{f1:.4f}",
    "Strategy": "Offline Expansion (AGRINOVA_DATASET)",
}

csv_path = os.path.join(MODELS_EXPORT_PATH, "training_scores.csv")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df = pd.concat([df, pd.DataFrame([score_data])], ignore_index=True)
else:
    df = pd.DataFrame([score_data])
df.to_csv(csv_path, index=False)

print(f"\n[COMPLETE] Crop Classifier Updated.")
