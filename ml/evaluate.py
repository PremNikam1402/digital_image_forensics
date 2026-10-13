"""Model Evaluation and Benchmark Reporting Script.

Loads trained model and tests it against specified authentic/tampered directories,
generating confusion matrix plots and academic benchmark summaries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.dataset import load_dataset_from_directory, extract_dataset_features


def evaluate_model(
    model_path: Path,
    test_dir: Path
) -> Dict[str, Any]:
    """Evaluate saved model on a directory of test images."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    print(f"[*] Loading model from: {model_path}")
    clf = joblib.load(model_path)

    print(f"[*] Scanning test images in: {test_dir}")
    image_paths, labels = load_dataset_from_directory(test_dir)
    if not image_paths:
        raise ValueError(f"No valid images found in {test_dir}")

    print(f"[*] Extracting forensic features for {len(image_paths)} images...")
    X, y = extract_dataset_features(image_paths, labels)

    print("[*] Generating predictions...")
    y_pred = clf.predict(X)

    acc = float(accuracy_score(y, y_pred))
    prec = float(precision_score(y, y_pred, zero_division=0))
    rec = float(recall_score(y, y_pred, zero_division=0))
    f1 = float(f1_score(y, y_pred, zero_division=0))
    cm = confusion_matrix(y, y_pred)

    print("\n========== FORENSIC EVALUATION REPORT ==========")
    print(f"Samples Tested:   {len(y)}")
    print(f"Accuracy:         {acc * 100:.2f}%")
    print(f"Precision:        {prec * 100:.2f}%")
    print(f"Recall:           {rec * 100:.2f}%")
    print(f"F1-Score:         {f1 * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=["Authentic", "Tampered"], zero_division=0))
    print("================================================")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm.tolist()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Forensic Tampering Model")
    parser.add_argument("--model-path", type=str, default="models/random_forest_forensic.joblib")
    parser.add_argument("--test-dir", type=str, default="data/sample")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    evaluate_model(
        model_path=project_root / args.model_path,
        test_dir=project_root / args.test_dir
    )
