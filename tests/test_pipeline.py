"""End-to-end integration tests for ML training, inference, and reporting."""

import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
import pytest

from ml.dataset import create_sample_dataset
from ml.train import train_forensic_classifier
from ml.predict import ForensicPredictor
from forensic.preprocessing import get_image_properties
from utils.report_generator import generate_forensic_report


def test_end_to_end_pipeline():
    """Verify entire workflow from dataset generation to training, prediction, and PDF generation."""
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        sample_dir = tmp_dir / "sample"
        model_path = tmp_dir / "model.joblib"
        metrics_path = tmp_dir / "metrics.json"

        # 1. Generate sample dataset
        auth_files, tamp_files = create_sample_dataset(sample_dir, num_samples=8)
        assert len(auth_files) == 8
        assert len(tamp_files) == 8

        # 2. Train Random Forest model
        metrics = train_forensic_classifier(
            data_dir=sample_dir,
            output_model_path=model_path,
            metrics_path=metrics_path,
            test_size=0.25,
            n_estimators=30,
        )
        assert model_path.exists()
        assert metrics_path.exists()
        assert "accuracy" in metrics
        assert "confusion_matrix" in metrics

        # 3. Predict on unseen sample using trained model
        predictor = ForensicPredictor(model_path=model_path, metrics_path=metrics_path)
        assert predictor.model_loaded is True

        test_img = Image.open(auth_files[0])
        pred_res = predictor.predict(test_img, image_path=auth_files[0])
        
        assert pred_res["prediction"] in ("LIKELY AUTHENTIC", "LIKELY MANIPULATED")
        assert 0.0 <= pred_res["confidence"] <= 1.0
        assert "authentic" in pred_res["probabilities"]
        assert "tampered" in pred_res["probabilities"]

        # 4. Generate PDF & Markdown forensic report
        props = get_image_properties(test_img, filename="test_sample.jpg")
        report = generate_forensic_report(pred_res, props)

        assert "CASE-" in report["case_id"]
        assert len(report["markdown"]) > 100
        assert len(report["pdf_bytes"]) > 1000
        assert report["pdf_bytes"][:4] == b"%PDF"
