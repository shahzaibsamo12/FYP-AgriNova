"""
AgriNova — Disease Detection Training Script
Stage 2: Rice & Cotton Disease Models

Usage:
    python train_disease.py                    # train all backbones, both crops
    python train_disease.py --crop rice        # rice only
    python train_disease.py --crop cotton      # cotton only
    python train_disease.py --backbone EfficientNetV2M  # single backbone
    python train_disease.py --no-aug           # use originals only (diagnostic)

Outputs (inside disease_detection/models/):
    rice_disease_model.keras
    rice_model_labels.pkl
    cotton_disease_model.keras
    cotton_model_labels.pkl

Outputs (inside reports/):
    agrinova_model_comparison.csv
    agrinova_model_comparison.xlsx
    <Crop>_<Model>_cm.png
    <Crop>_<Model>_curves.png
"""

import os
import sys
import time
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.applications import (
    EfficientNetV2B0, EfficientNetV2B3,
    EfficientNetV2M, EfficientNetV2L,
    MobileNetV3Large, ResNet50, DenseNet121
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Paths — FIXED FOR YOUR ACTUAL DATASET
# ─────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DATASETS_DIR = os.path.join(PROJECT_ROOT, 'datasets')

# ---------------------
# RICE PATHS
# ---------------------
RICE_BASE = os.path.join(
    DATASETS_DIR,
    'rice',
    'Rice Leaf Bacterial and Fungal Disease Dataset'
)

RICE_ORIG = os.path.join(RICE_BASE, 'Original')

RICE_AUG = os.path.join(RICE_BASE, 'Augmented')

# ---------------------
# COTTON PATHS
# ---------------------
COTTON_BASE = os.path.join(
    DATASETS_DIR,
    'cotton',
    'SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection'
)

COTTON_ORIG = os.path.join(COTTON_BASE, 'Original Dataset')

COTTON_AUG = os.path.join(
    COTTON_BASE,
    'Augmented Dataset',
    'Augmented Dataset'
)

# ---------------------
# OUTPUT PATHS
# ---------------------
MODELS_PATH = os.path.join(PROJECT_ROOT, 'models')
REPORTS_PATH = os.path.join(PROJECT_ROOT, 'reports')

os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(REPORTS_PATH, exist_ok=True)

# ─────────────────────────────────────────────────────────────
#  Hyperparameters
# ─────────────────────────────────────────────────────────────
IMAGE_SIZE      = (224, 224)
BATCH_SIZE      = 32
EPOCHS_STAGE1   = 20
EPOCHS_STAGE2   = 15
UNFREEZE_LAYERS = 40
SEED            = 42

ALL_BACKBONES   = [
    'EfficientNetV2B0',
    'EfficientNetV2B3',
    'EfficientNetV2M',
    'EfficientNetV2L',
    'MobileNetV3Large',
    'ResNet50',
    'DenseNet121',
]

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ─────────────────────────────────────────────────────────────
#  Mixed precision (GPU)
# ─────────────────────────────────────────────────────────────
# Disable mixed precision to avoid NaN issues
# try:
#     tf.keras.mixed_precision.set_global_policy('mixed_float16')
#     print('[INFO] Mixed precision: float16 enabled')
# except Exception:
#     print('[INFO] Mixed precision: not available')
print('[INFO] Using float32 precision (mixed precision disabled for stability)')


# ─────────────────────────────────────────────────────────────
#  Dataset utilities
# ─────────────────────────────────────────────────────────────
def compute_class_weights_manual(labels):
    """
    Manually compute balanced class weights without using sklearn
    """
    unique_classes = np.unique(labels)
    n_samples = len(labels)
    n_classes = len(unique_classes)
    
    class_weights = {}
    for cls in unique_classes:
        n_cls = np.sum(labels == cls)
        weight = n_samples / (n_classes * n_cls)
        # Clip weights to avoid extreme values
        weight = np.clip(weight, 0.5, 10.0)
        class_weights[int(cls)] = float(weight)
    
    return class_weights


def collect_image_paths(original_root, aug_root, use_aug=True):
    """
    Strict isolation split:
      - Original images → 70% train / 15% val / 15% test (seeded)
      - Augmented images → 100% added to train only
    Returns train/val/test path+label lists, class names, class weights.
    """

    if not os.path.exists(original_root):
        raise FileNotFoundError(f'Original data not found: {original_root}')

    class_names = sorted([
        d for d in os.listdir(original_root)
        if os.path.isdir(os.path.join(original_root, d))
    ])

    class_to_id = {c: i for i, c in enumerate(class_names)}

    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []

    rng = np.random.RandomState(SEED)

    # -----------------------------
    # Load ORIGINAL dataset
    # -----------------------------
    for cls_name in class_names:
        cls_path = os.path.join(original_root, cls_name)

        imgs = []
        for root, _, files in os.walk(cls_path):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    imgs.append(os.path.join(root, f))

        if len(imgs) == 0:
            print(f'  Warning: No images found in {cls_path}')
            continue

        rng.shuffle(imgs)

        lbl = class_to_id[cls_name]
        n = len(imgs)

        n_tr = int(n * 0.70)
        n_v = int(n * 0.15)

        train_paths += imgs[:n_tr]
        train_labels += [lbl] * n_tr

        val_paths += imgs[n_tr:n_tr + n_v]
        val_labels += [lbl] * n_v

        test_paths += imgs[n_tr + n_v:]
        test_labels += [lbl] * (n - n_tr - n_v)

    # -----------------------------
    # Add AUGMENTED data (train only)
    # -----------------------------
    if use_aug and aug_root and os.path.exists(aug_root):
        print(f'  Loading augmented data from: {aug_root}')
        for cls_name in class_names:
            aug_cls = os.path.join(aug_root, cls_name)

            if not os.path.exists(aug_cls):
                print(f'  Warning: No augmented data for {cls_name}')
                continue

            lbl = class_to_id[cls_name]

            for root, _, files in os.walk(aug_cls):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        train_paths.append(os.path.join(root, f))
                        train_labels.append(lbl)

    # =========================================================
    # FIXED CLASS WEIGHT SECTION - Manual calculation
    # =========================================================
    
    # Convert to numpy arrays
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)
    test_labels = np.array(test_labels)
    
    # Compute class weights manually (avoiding sklearn issues)
    cw_dict = compute_class_weights_manual(train_labels)

    # -----------------------------
    # PRINT STATS
    # -----------------------------
    print(f'  Classes  : {class_names}')
    print(f'  Class IDs: {class_to_id}')
    print(f'  Train    : {len(train_paths)} images')
    print(f'  Val      : {len(val_paths)} images')
    print(f'  Test     : {len(test_paths)} images (originals only)')
    print(f'  Class distribution:')
    for i, cls_name in enumerate(class_names):
        train_count = np.sum(train_labels == i)
        val_count = np.sum(val_labels == i)
        test_count = np.sum(test_labels == i)
        print(f'    {cls_name}: Train={train_count}, Val={val_count}, Test={test_count}')
    print(f'  Class weights: {cw_dict}')

    return (
        train_paths, train_labels,
        val_paths, val_labels,
        test_paths, test_labels,
        class_names, cw_dict
    )


# ─────────────────────────────────────────────────────────────
#  TF Dataset builder with fixed augmentation
# ─────────────────────────────────────────────────────────────
def load_and_preprocess_image(path, label, augment=False):
    """Load, decode, and preprocess a single image"""
    try:
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1]
        
        if augment:
            # Random horizontal flip
            image = tf.image.random_flip_left_right(image)
            # Random brightness (small adjustment)
            image = tf.image.random_brightness(image, max_delta=0.1)
            # Random contrast (small adjustment)
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
            # Random saturation (small adjustment)
            image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
            # Ensure values stay in [0,1]
            image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, label
    except Exception as e:
        print(f'Error loading image {path}: {e}')
        # Return a blank image as fallback
        blank_image = tf.zeros((*IMAGE_SIZE, 3), dtype=tf.float32)
        return blank_image, label


def build_tf_dataset(image_paths, labels, shuffle=False, augment=False):
    """Build TensorFlow dataset from image paths and labels"""
    if len(image_paths) == 0:
        raise ValueError("No images provided to build dataset")
    
    # Convert to tensorflow constants for better performance
    paths_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((paths_ds, labels_ds))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(image_paths), 1000))
    
    dataset = dataset.map(
        lambda x, y: load_and_preprocess_image(x, y, augment),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


# ─────────────────────────────────────────────────────────────
#  Model builder
# ─────────────────────────────────────────────────────────────
BACKBONE_MAP = {
    'EfficientNetV2B0': EfficientNetV2B0,
    'EfficientNetV2B3': EfficientNetV2B3,
    'EfficientNetV2M': EfficientNetV2M,
    'EfficientNetV2L': EfficientNetV2L,
    'MobileNetV3Large': MobileNetV3Large,
    'ResNet50': ResNet50,
    'DenseNet121': DenseNet121,
}


def build_model(backbone_name, num_classes, dropout=0.5, l2=1e-4, dense=512):
    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    
    # Normalize to [0,1] first
    x = layers.Rescaling(1.0)(inputs)
    
    # Then apply backbone-specific preprocessing
    if backbone_name.startswith('EfficientNet'):
        # EfficientNet expects [0, 255]
        x = layers.Rescaling(255.0)(x)
    elif backbone_name in ['ResNet50', 'DenseNet121']:
        # These expect [-1, 1]
        x = layers.Rescaling(2.0, offset=-1.0)(x)
    # MobileNetV3 expects [0, 1] which we already have

    base = BACKBONE_MAP[backbone_name](
        weights='imagenet', include_top=False, input_tensor=x
    )
    base.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(dense, activation='relu',
                     kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(dropout * 0.5)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, out, name=backbone_name), base


def get_callbacks(save_path, patience_es=8, patience_lr=4):
    return [
        callbacks.ModelCheckpoint(
            save_path, monitor='val_accuracy',
            save_best_only=True, verbose=1, mode='max'
        ),
        callbacks.EarlyStopping(
            monitor='val_accuracy', patience=patience_es,
            restore_best_weights=True, verbose=1, mode='max'
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=patience_lr, min_lr=1e-7, verbose=1
        ),
    ]


# ─────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────
def evaluate_model(model, test_ds, class_names, model_label,
                   crop_name, hist1, hist2=None):
    y_true, y_pred, y_prob = [], [], []
    for imgs, lbls in test_ds:
        probs = model.predict(imgs, verbose=0)
        y_true.extend(lbls.numpy().tolist())
        y_pred.extend(np.argmax(probs, axis=1).tolist())
        y_prob.extend(probs.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc  = float(np.mean(y_true == y_pred))
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    try:
        y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
        roc   = roc_auc_score(y_bin, y_prob, multi_class='ovr', average='macro')
    except Exception as e:
        print(f'Warning: Could not compute ROC-AUC: {e}')
        roc = None

    print(f'\n{"="*55}')
    print(f'  {model_label} | {crop_name}')
    print(f'{"="*55}')
    print(f'  Accuracy  : {acc:.4f}')
    print(f'  Precision : {prec:.4f}')
    print(f'  Recall    : {rec:.4f}')
    print(f'  F1-Score  : {f1:.4f}')
    if roc:
        print(f'  ROC-AUC   : {roc:.4f}')
    print()
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f'{crop_name} — {model_label}\nConfusion Matrix (counts)')
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
    plt.setp(axes[0].get_xticklabels(), rotation=40, ha='right', fontsize=8)

    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_n, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title(f'{crop_name} — {model_label}\nConfusion Matrix (normalised)')
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
    plt.setp(axes[1].get_xticklabels(), rotation=40, ha='right', fontsize=8)

    plt.tight_layout()
    cm_path = os.path.join(REPORTS_PATH, f'{crop_name}_{model_label}_cm.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Training curves
    def merge(h1, h2):
        if h2 is None:
            return h1.history
        # Filter out nan values
        history = {}
        for k in h1.history:
            values = h1.history[k]
            if h2 and k in h2.history:
                values = values + h2.history[k]
            # Remove nan values
            values = [v for v in values if not np.isnan(v)]
            history[k] = values
        return history

    hist = merge(hist1, hist2)
    
    if hist.get('accuracy') and hist.get('val_accuracy'):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].plot(hist['accuracy'], label='Train Acc')
        axes[0].plot(hist['val_accuracy'], label='Val Acc')
        axes[0].set_title(f'{crop_name} — {model_label}\nAccuracy')
        axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(hist.get('loss', []), label='Train Loss')
        axes[1].plot(hist.get('val_loss', []), label='Val Loss')
        axes[1].set_title(f'{crop_name} — {model_label}\nLoss')
        axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        curve_path = os.path.join(REPORTS_PATH, f'{crop_name}_{model_label}_curves.png')
        plt.savefig(curve_path, dpi=150, bbox_inches='tight')
        plt.close()

    val_accs = hist.get('val_accuracy', [0])
    val_losses = hist.get('val_loss', [999])

    return {
        'Model':            model_label,
        'Dataset':          crop_name,
        'Accuracy':         round(acc, 4),
        'Precision':        round(prec, 4),
        'Recall':           round(rec, 4),
        'F1_Score':         round(f1, 4),
        'ROC_AUC':          round(roc, 4) if roc else None,
        'Val_Accuracy':     round(max(val_accs), 4) if val_accs else 0,
        'Val_Loss':         round(min(val_losses), 4) if val_losses else 999,
    }


# ─────────────────────────────────────────────────────────────
#  Single model training run
# ─────────────────────────────────────────────────────────────
def train_one_model(backbone_name, crop_name, orig_root, aug_root, use_aug):
    print(f'\n{"#"*60}')
    print(f'  Training: {backbone_name} | {crop_name}')
    print(f'{"#"*60}')

    t0 = time.time()

    try:
        (tr_p, tr_l, va_p, va_l,
         te_p, te_l, cls_names, cw) = collect_image_paths(orig_root, aug_root, use_aug)
    except Exception as e:
        print(f'Error collecting image paths: {e}')
        raise

    if len(tr_p) == 0:
        raise ValueError(f'No training images found for {crop_name}')

    print(f'  Building datasets...')
    train_ds = build_tf_dataset(tr_p, tr_l, shuffle=True, augment=True)
    val_ds   = build_tf_dataset(va_p, va_l, shuffle=False, augment=False)
    test_ds  = build_tf_dataset(te_p, te_l, shuffle=False, augment=False)

    ckpt = os.path.join(MODELS_PATH, f'_tmp_{crop_name}_{backbone_name}.keras')
    model, base = build_model(backbone_name, len(cls_names))

    # Stage 1: Frozen warm-up with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),  # Reduced learning rate
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f'\n[Stage 1] Frozen warm-up ({EPOCHS_STAGE1} epochs max)')
    h1 = model.fit(
        train_ds, epochs=EPOCHS_STAGE1, validation_data=val_ds,
        class_weight=cw,
        callbacks=get_callbacks(ckpt, patience_es=6, patience_lr=3),
        verbose=1
    )

    # Stage 2: Fine-tune top layers
    base.trainable = True
    for layer in base.layers[:-UNFREEZE_LAYERS]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),  # Even lower learning rate for fine-tuning
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f'\n[Stage 2] Fine-tune top-{UNFREEZE_LAYERS} layers ({EPOCHS_STAGE2} epochs max)')
    h2 = model.fit(
        train_ds, epochs=EPOCHS_STAGE2, validation_data=val_ds,
        class_weight=cw,
        callbacks=get_callbacks(ckpt, patience_es=5, patience_lr=3),
        verbose=1
    )

    elapsed = time.time() - t0
    metrics = evaluate_model(model, test_ds, cls_names,
                             backbone_name, crop_name, h1, h2)
    metrics['Training_Time_s'] = round(elapsed, 1)

    # Clean up temp checkpoint
    if os.path.exists(ckpt):
        os.remove(ckpt)

    return model, cls_names, metrics


# ─────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='AgriNova Disease Model Trainer')
    parser.add_argument('--crop',      choices=['rice', 'cotton', 'both'], default='both')
    parser.add_argument('--backbone',  choices=ALL_BACKBONES + ['all'],    default='all')
    parser.add_argument('--no-aug',    action='store_true',
                        help='Disable offline augmented data (diagnostic mode)')
    args = parser.parse_args()

    use_aug   = not args.no_aug
    backbones = ALL_BACKBONES if args.backbone == 'all' else [args.backbone]
    crops     = ['rice', 'cotton'] if args.crop == 'both' else [args.crop]

    print(f'[CONFIG] Crops: {crops}')
    print(f'[CONFIG] Backbones: {backbones}')
    print(f'[CONFIG] Use augmented data: {use_aug}')
    print(f'[CONFIG] GPU devices: {tf.config.list_physical_devices("GPU")}')

    all_metrics = []
    best_per_crop = {}

    for crop in crops:
        orig_root = RICE_ORIG   if crop == 'rice' else COTTON_ORIG
        aug_root  = RICE_AUG    if crop == 'rice' else COTTON_AUG
        crop_label = crop.capitalize()

        print(f'\n{"="*60}')
        print(f'Processing {crop_label} dataset')
        print(f'Original path: {orig_root}')
        print(f'Augmented path: {aug_root}')
        print(f'{"="*60}')

        # Verify paths exist
        if not os.path.exists(orig_root):
            print(f'ERROR: Original path does not exist: {orig_root}')
            continue

        crop_results = []

        for bb in backbones:
            try:
                model, cls_names, metrics = train_one_model(
                    bb, crop_label, orig_root, aug_root, use_aug
                )
                crop_results.append(metrics)
                all_metrics.append(metrics)

                # Track best
                if (crop_label not in best_per_crop or
                        metrics['Val_Accuracy'] > best_per_crop[crop_label]['val_acc']):
                    best_per_crop[crop_label] = {
                        'model':     model,
                        'cls_names': cls_names,
                        'backbone':  bb,
                        'val_acc':   metrics['Val_Accuracy'],
                    }
            except Exception as e:
                print(f'Error training {bb} on {crop_label}: {e}')
                import traceback
                traceback.print_exc()
                continue

            tf.keras.backend.clear_session()

        # Print crop comparison table
        if crop_results:
            cdf = pd.DataFrame(crop_results).sort_values('Val_Accuracy', ascending=False)
            print(f'\n{crop_label} Architecture Comparison:')
            print(cdf[['Model', 'Accuracy', 'F1_Score', 'Val_Accuracy',
                        'Training_Time_s']].to_string(index=False))

    # ── Save best models ──────────────────────────────────────
    for crop_label, info in best_per_crop.items():
        safe_crop = crop_label.lower()
        model_path = os.path.join(MODELS_PATH, f'{safe_crop}_disease_model.keras')
        label_path = os.path.join(MODELS_PATH, f'{safe_crop}_model_labels.pkl')

        # Save the best model
        info['model'].save(model_path)
        with open(label_path, 'wb') as f:
            pickle.dump(info['cls_names'], f)

        print(f'\n[SAVED] {model_path}')
        print(f'[SAVED] {label_path}')
        print(f'  Best backbone : {info["backbone"]}')
        print(f'  Val accuracy  : {info["val_acc"]:.4f}')

    # ── Save reports ──────────────────────────────────────────
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        cols = ['Model', 'Dataset', 'Accuracy', 'Precision', 'Recall',
                'F1_Score', 'ROC_AUC', 'Val_Accuracy', 'Val_Loss', 'Training_Time_s']
        df = df[[c for c in cols if c in df.columns]]

        csv_path  = os.path.join(REPORTS_PATH, 'agrinova_model_comparison.csv')
        xlsx_path = os.path.join(REPORTS_PATH, 'agrinova_model_comparison.xlsx')

        df.to_csv(csv_path, index=False)

        try:
            with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='All Models', index=False)
                for crop_label in set([m['Dataset'] for m in all_metrics]):
                    subset = df[df['Dataset'] == crop_label]
                    if not subset.empty:
                        subset.to_excel(writer, sheet_name=f'{crop_label} Comparison', index=False)
            print(f'\n[SAVED] {xlsx_path}')
        except Exception as e:
            print(f'Warning: Could not save Excel file: {e}')

        print(f'\n[SAVED] {csv_path}')

    print('\n[COMPLETE] AgriNova Disease Training finished.')


if __name__ == '__main__':
    main()