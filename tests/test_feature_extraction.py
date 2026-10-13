"""Tests for unified 24-dimensional feature extraction pipeline."""

from PIL import Image
import numpy as np
import pytest

from forensic.feature_extraction import extract_all_features, FEATURE_NAMES


def test_extract_all_features_structure():
    """Verify that feature extraction yields correct vector length and valid numbers."""
    img = Image.new("RGB", (120, 100), color=(100, 120, 140))
    vec, feat_dict, full_results = extract_all_features(img, include_metadata=False)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (24,)
    assert len(FEATURE_NAMES) == 24
    assert len(feat_dict) == 24

    # Assert no NaNs or Infs exist in feature vector
    assert not np.isnan(vec).any(), "Feature vector contains NaN values"
    assert not np.isinf(vec).any(), "Feature vector contains Inf values"

    # Verify presence of all expected sub-results
    assert "ela" in full_results
    assert "noise" in full_results
    assert "histogram" in full_results
    assert "copy_move" in full_results
    assert "metadata" in full_results
