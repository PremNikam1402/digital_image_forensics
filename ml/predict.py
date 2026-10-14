"""Forensic Inference and Calibrated Tampering Prediction Module.

Loads serialized Random Forest model, runs unified feature extraction,
and returns objective, non-fabricated probability scores and classification verdicts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Union, Optional
from PIL import Image
import numpy as np
import joblib

from forensic.feature_extraction import extract_all_features, FEATURE_NAMES, FEATURE_DESCRIPTIONS


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "random_forest_forensic.joblib"
DEFAULT_METRICS_PATH = Path(__file__).resolve().parent.parent / "models" / "model_metrics.json"


class ForensicPredictor:
    """Inference engine for image tampering detection."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        metrics_path: Optional[Union[str, Path]] = None
    ):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.metrics_path = Path(metrics_path) if metrics_path else DEFAULT_METRICS_PATH
        self.model = None
        self.metrics = None
        self._load_model()

    @property
    def model_loaded(self) -> bool:
        """Check if machine learning model is loaded and available."""
        return self.model is not None


    def _load_model(self) -> None:
        """Load trained Random Forest model and associated test metrics if available."""
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
            except Exception as e:
                print(f"[!] Warning: Could not load model from {self.model_path}: {e}")
                self.model = None

        if self.metrics_path.exists():
            try:
                with open(self.metrics_path, "r", encoding="utf-8") as f:
                    self.metrics = json.load(f)
            except Exception:
                self.metrics = None

    def predict(
        self,
        image: Union[Image.Image, np.ndarray],
        image_path: Optional[Union[str, Path]] = None,
        copy_move_algo: str = "SIFT",
        ela_quality: int = 90,
        lowe_ratio: float = 0.75,
        min_spatial_distance: float = 35.0,
    ) -> Dict[str, Any]:
        """Perform full forensic analysis and ML tampering prediction.
        
        Parameters:
            image: PIL Image or RGB numpy array
            image_path: Path to disk file (for EXIF extraction)
            copy_move_algo: 'SIFT' or 'ORB'
            ela_quality: ELA JPEG quality level
            lowe_ratio: Lowe's ratio test threshold (default 0.75)
            min_spatial_distance: Minimum pixel distance between matched keypoints
            
        Returns:
            Dictionary containing:
              - prediction: 'LIKELY AUTHENTIC' or 'LIKELY MANIPULATED'
              - confidence: Actual calculated probability (0.0 to 1.0)
              - probabilities: {'authentic': float, 'tampered': float}
              - feature_vector: 24-dim numpy array
              - features: dict mapping feature names to values
              - full_results: nested analytical results from all modules
              - top_features: top active features
              - model_loaded: bool
        """
        # Step 1: Extract unified 24-dimensional feature vector
        vec, features, full_results = extract_all_features(
            image=image,
            image_path=image_path,
            include_metadata=True,
            copy_move_algo=copy_move_algo,
            ela_quality=ela_quality,
            lowe_ratio=lowe_ratio,
            min_spatial_distance=min_spatial_distance,
        )

        # Step 2: Supervised ML Inference
        if self.model is not None:
            # Reshape for single sample prediction
            X_sample = vec.reshape(1, -1)
            
            try:
                probs = self.model.predict_proba(X_sample)[0]
                # Assuming class 0 is authentic, class 1 is tampered
                prob_auth = float(probs[0])
                prob_tamp = float(probs[1]) if len(probs) > 1 else 1.0 - prob_auth
            except Exception:
                pred_label = self.model.predict(X_sample)[0]
                prob_tamp = 1.0 if pred_label == 1 else 0.0
                prob_auth = 1.0 - prob_tamp

            is_tampered = prob_tamp >= 0.50
            confidence = prob_tamp if is_tampered else prob_auth
            verdict = "LIKELY MANIPULATED" if is_tampered else "LIKELY AUTHENTIC"
            source = "Supervised Random Forest Classifier"

        else:
            # Fallback heuristic if ML model is not yet trained
            # Uses conservative forensic rule integration
            tamp_score = 0.0
            
            # Copy-Move inliers is decisive physical evidence
            if full_results["copy_move"]["results"]["evidence_found"]:
                tamp_score += 0.50
            # ELA inconsistency
            if full_results["ela"]["stats"]["suspicious_ela"]:
                tamp_score += 0.25
            # Noise inconsistency
            if full_results["noise"]["stats"]["suspicious_noise"]:
                tamp_score += 0.20
            # Comb artifact
            if full_results["histogram"]["comb_effect_detected"]:
                tamp_score += 0.15

            tamp_score = min(0.98, max(0.05, tamp_score))
            prob_tamp = float(tamp_score)
            prob_auth = float(1.0 - tamp_score)
            is_tampered = prob_tamp >= 0.50
            confidence = prob_tamp if is_tampered else prob_auth
            verdict = "LIKELY MANIPULATED" if is_tampered else "LIKELY AUTHENTIC"
            source = "Heuristic Forensic Rule Integrator (Model not yet trained)"

        # Step 3: Identify Top Relevant Contributing Features
        top_features = []
        if self.metrics and "feature_importances" in self.metrics:
            for item in self.metrics["feature_importances"][:6]:
                fname = item["feature"]
                top_features.append({
                    "name": fname,
                    "value": round(features.get(fname, 0.0), 3),
                    "description": FEATURE_DESCRIPTIONS.get(fname, ""),
                    "global_importance": item["importance"]
                })
        else:
            # Fallback to key forensic features
            for fname in ["cm_inliers_count", "ela_block_ratio", "noise_inconsistency_ratio", "hist_comb_ratio"]:
                top_features.append({
                    "name": fname,
                    "value": round(features.get(fname, 0.0), 3),
                    "description": FEATURE_DESCRIPTIONS.get(fname, ""),
                    "global_importance": 0.25
                })

        return {
            "prediction": verdict,
            "is_tampered": is_tampered,
            "confidence": round(confidence, 4),
            "confidence_percent": round(confidence * 100, 2),
            "probabilities": {
                "authentic": round(prob_auth, 4),
                "tampered": round(prob_tamp, 4),
            },
            "source": source,
            "model_loaded": self.model is not None,
            "feature_vector": vec,
            "features": features,
            "top_features": top_features,
            "full_results": full_results,
            "metrics": self.metrics,
        }
