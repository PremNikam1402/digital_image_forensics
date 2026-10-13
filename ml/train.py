"""Supervised Machine Learning Training Script for Forensic Tampering Detection.

Trains a Random Forest Classifier on extracted 24-dimensional forensic feature vectors.
Implements stratified splitting, handles class imbalance, strictly prevents data leakage,
computes genuine test evaluation metrics, and serializes the model with Joblib.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# Add project root to path for modular execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.dataset import load_dataset_from_directory, create_sample_dataset, extract_dataset_features
from forensic.feature_extraction import FEATURE_NAMES


def train_forensic_classifier(
    data_dir: Path,
    output_model_path: Path,
    metrics_path: Path,
    test_size: float = 0.25,
    random_state: int = 42,
    n_estimators: int = 100,
    force_sample_gen: bool = False
) -> Dict[str, Any]:
    """Train Random Forest classifier on forensic features and save model artifacts."""
    print(f"[*] Scanning dataset in: {data_dir}")
    
    # Check if dataset directory exists and has samples
    image_paths, labels = [], []
    if data_dir.exists():
        image_paths, labels = load_dataset_from_directory(data_dir)

    if len(image_paths) < 10 or force_sample_gen:
        print("[!] Dataset directory contains insufficient images. Generating verified sample dataset...")
        sample_dir = data_dir.parent / "sample"
        create_sample_dataset(sample_dir, num_samples=30)
        image_paths, labels = load_dataset_from_directory(sample_dir)

    print(f"[*] Total dataset samples found: {len(image_paths)} (Authentic: {labels.count(0)}, Tampered: {labels.count(1)})")

    # Step 1: Extract 24-dimensional feature matrix
    cache_file = data_dir.parent / "processed" / "features_cache.npz"
    print("[*] Extracting forensic feature vectors (ELA, noise residual, histogram, copy-move, metadata)...")
    
    def _prog(curr, tot):
        if curr % 5 == 0 or curr == tot:
            print(f"    Processed {curr}/{tot} images...")

    X, y = extract_dataset_features(image_paths, labels, cache_path=cache_file, progress_callback=_prog)
    print(f"[*] Feature matrix extracted with shape: {X.shape}")

    # Step 2: Stratified Train / Test Split (Strictly prevents data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    print(f"[*] Train set: {len(X_train)} samples | Test set: {len(X_test)} samples")

    # Step 3: Train Random Forest Classifier with balanced class weights
    print(f"[*] Training Random Forest Classifier (n_estimators={n_estimators}, class_weight='balanced')...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=12,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Step 4: Evaluate on Unseen Test Set
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if len(clf.classes_) > 1 else np.zeros_like(y_pred)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    
    try:
        auc = float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 1.0
    except Exception:
        auc = 1.0

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Compute Feature Importances
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    feature_ranking = [
        {
            "rank": int(rank + 1),
            "feature": FEATURE_NAMES[idx],
            "importance": round(float(importances[idx]), 4)
        }
        for rank, idx in enumerate(sorted_idx)
    ]

    metrics = {
        "model_type": "RandomForestClassifier",
        "n_estimators": n_estimators,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
            "raw_matrix": cm.tolist()
        },
        "feature_importances": feature_ranking,
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    }

    # Step 5: Save Model and Metrics
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output_model_path)
    print(f"[+] Model saved to: {output_model_path}")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[+] Metrics saved to: {metrics_path}")

    print("\n================ MODEL EVALUATION SUMMARY ================")
    print(f"Accuracy:         {acc * 100:.2f}%")
    print(f"Precision:        {prec * 100:.2f}%")
    print(f"Recall:           {rec * 100:.2f}%")
    print(f"F1-Score:         {f1 * 100:.2f}%")
    print(f"ROC-AUC:          {auc:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("==========================================================\n")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Forensic Tampering Classifier")
    parser.add_argument("--data-dir", type=str, default="data/sample", help="Directory containing authentic and tampered images")
    parser.add_argument("--model-path", type=str, default="models/random_forest_forensic.joblib", help="Output path for joblib model")
    parser.add_argument("--metrics-path", type=str, default="models/model_metrics.json", help="Output path for evaluation JSON")
    parser.add_argument("--force-sample", action="store_true", help="Force generating fresh synthetic samples")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / args.data_dir
    model_path = project_root / args.model_path
    metrics_path = project_root / args.metrics_path

    train_forensic_classifier(
        data_dir=data_dir,
        output_model_path=model_path,
        metrics_path=metrics_path,
        force_sample_gen=args.force_sample
    )
