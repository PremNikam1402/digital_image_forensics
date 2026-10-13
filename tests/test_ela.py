"""Tests for Error Level Analysis (ELA) computation and metrics."""

from PIL import Image
import numpy as np
import pytest

from forensic.ela import perform_ela


def test_perform_ela_basic():
    """Verify basic ELA calculation on a synthetic image."""
    img = Image.new("RGB", (128, 128), color=(100, 150, 200))
    ela_pil, diff, stats = perform_ela(img, quality=90, scale=15)
    
    assert ela_pil is not None
    assert ela_pil.size == (128, 128)
    assert diff.shape == (128, 128, 3)
    assert "mean_error" in stats
    assert "max_error" in stats
    assert "percentile_95" in stats
    assert stats["quality_level"] == 90
    assert stats["scale_factor"] == 15
    assert stats["mean_error"] >= 0.0


def test_perform_ela_varying_quality():
    """Verify that different compression qualities produce different error levels."""
    # Textured pattern
    arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    img = Image.fromarray(arr)

    _, _, stats_q70 = perform_ela(img, quality=70)
    _, _, stats_q95 = perform_ela(img, quality=95)

    # Lower quality (70) should have greater compression degradation than quality (95)
    assert stats_q70["mean_error"] >= stats_q95["mean_error"]


def test_perform_ela_numpy_input():
    """Verify ELA functions properly when passed a numpy array directly."""
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[20:40, 20:40] = 255
    ela_pil, diff, stats = perform_ela(arr, quality=90)
    assert diff.shape == (64, 64, 3)
