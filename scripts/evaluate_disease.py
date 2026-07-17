"""
AgriNova — Evaluation Script
Loads saved rice_disease_model.keras and cotton_disease_model.keras
and runs the full evaluation suite on a test directory.

Usage:
    python evaluate_disease.py --crop rice   --test-dir path/to/rice/test
    python evaluate_disease.py --crop cotton --test-dir path/to/cotton/test
    python evaluate_disease.py --all                         # evaluate both from default paths
"""

import os
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_PATH  = os.path.join(PROJECT_ROOT, 'disease_detection', 'models')
REPORTS_PATH = os.path.join(PROJECT_ROOT, 'reports')
DATA_ROOT    = os.path.join(PROJECT_ROOT, 'datasets')

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


def load_model_and_labels(crop_name):
    safe = crop_name.lower()
    model_path = os.path.join(MODELS_PATH, f'{safe}_disease_model.keras')
    label_path = os.path.join(MODELS_PATH, f'{safe}_model_labels.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found: {model_path}\nTrain first with train_disease.py')
    if not os.path.exists(label_path):
        raise FileNotFoundError(f'Labels not found: {label_path}')

    model = tf.keras.models.load_model(model_path)
    with open(label_path, 'rb') as f:
        labels = pickle.load(f)
    print(f'[OK] Loaded {crop_name} model  ({len(labels)} classes)')
    return model, labels


def build_test_dataset(test_root, class_names):
    """Builds test dataset from a folder structure: test_root/class_name/image.jpg"""
    class_to_id = {c: i for i, c in enumerate(class_names)}
    paths, labels = [], []

    for cls_name in sorted(os.listdir(test_root)):
        cls_path = os.path.join(test_root, cls_name)
        if not os.path.isdir(cls_path):
            continue
        if cls_name not in class_to_id:
            print(f'  [WARN] Unknown class folder: {cls_name} — skipping')
            continue
        lbl = class_to_id[cls_name]
        for root, _, files in os.walk(cls_path):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    paths.append(os.path.join(root, f))
                    labels.append(lbl)

    def decode(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, IMAGE_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds, len(paths)


def run_evaluation(crop_name, test_root):
    print(f'\n{"="*60}')
    print(f'  Evaluating: {crop_name}')
    print(f'  Test root : {test_root}')
    print(f'{"="*60}')

    os.makedirs(REPORTS_PATH, exist_ok=True)

    model, class_names = load_model_and_labels(crop_name)
    test_ds, n_samples = build_test_dataset(test_root, class_names)

    if n_samples == 0:
        print('[ERROR] No test images found.')
        return None

    print(f'  Test samples: {n_samples}')

    # Inference
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
    except Exception:
        roc = None

    print(f'\n  Accuracy  : {acc:.4f}')
    print(f'  Precision : {prec:.4f}')
    print(f'  Recall    : {rec:.4f}')
    print(f'  F1-Score  : {f1:.4f}')
    if roc:
        print(f'  ROC-AUC   : {roc:.4f}')
    print('\n  Per-class Report:')
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f'{crop_name} Evaluation\nConfusion Matrix (counts)')
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
    plt.setp(axes[0].get_xticklabels(), rotation=40, ha='right', fontsize=8)

    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_n, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title(f'{crop_name} Evaluation\nConfusion Matrix (normalised)')
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
    plt.setp(axes[1].get_xticklabels(), rotation=40, ha='right', fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(REPORTS_PATH, f'{crop_name}_eval_cm.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [SAVED] {fig_path}')

    return {
        'Crop':         crop_name,
        'Test_Samples': n_samples,
        'Accuracy':     round(acc, 4),
        'Precision':    round(prec, 4),
        'Recall':       round(rec, 4),
        'F1_Score':     round(f1, 4),
        'ROC_AUC':      round(roc, 4) if roc else None,
    }


def main():
    parser = argparse.ArgumentParser(description='AgriNova Disease Evaluator')
    parser.add_argument('--crop',      choices=['rice', 'cotton', 'both'], default='both')
    parser.add_argument('--test-dir',  default=None,
                        help='Path to test directory (class-folder structure). '
                             'If omitted, uses default dataset test split.')
    args = parser.parse_args()

    crop_test_map = {
        'Rice': os.path.join(DATA_ROOT, 'rice', 'Rice Leaf Bacterial and Fungal Disease Dataset', 'Original'),
        'Cotton': os.path.join(DATA_ROOT, 'cotton', 'SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection', 'Original Dataset'),
    }
    if args.test_dir:
        # Use provided path for all crops
        crops = [args.crop.capitalize()] if args.crop != 'both' else ['Rice', 'Cotton']
        crop_test_map = {c: args.test_dir for c in crops}
    elif args.crop != 'both':
        crops = [args.crop.capitalize()]
    else:
        crops = ['Rice', 'Cotton']

    results = []
    for crop in crops:
        test_root = crop_test_map.get(crop)
        if not test_root:
            print(f'[SKIP] No test path for {crop}')
            continue
        r = run_evaluation(crop, test_root)
        if r:
            results.append(r)

    if results:
        df = pd.DataFrame(results)
        csv_path  = os.path.join(REPORTS_PATH, 'agrinova_evaluation_results.csv')
        xlsx_path = os.path.join(REPORTS_PATH, 'agrinova_evaluation_results.xlsx')
        df.to_csv(csv_path, index=False)
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Evaluation', index=False)
        print(f'\n[SAVED] {csv_path}')
        print(f'[SAVED] {xlsx_path}')
        print(df.to_string(index=False))


if __name__ == '__main__':
    main()
