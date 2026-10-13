"""Tests for noise analysis and local variance computation."""

from PIL import Image
import numpy as np
import pytest

from forensic.noise_analysis import perform_noise_analysis


def test_perform_noise_analysis_uniform():
    """Verify noise analysis on a synthetic noisy image."""
    np.random.seed(42)
    base = np.full((128, 128, 3), 128, dtype=np.uint8)
    noise = np.random.normal(0, 10, (128, 128, 3))
    noisy = np.clip(base + noise, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(noisy)
    vis, local_vars, stats = perform_noise_analysis(img, block_size=16)

    assert vis.shape == (128, 128)
    assert local_vars.shape == (8, 8)
    assert "global_noise_variance" in stats
    assert "snr_db" in stats
    assert "residual_kurtosis" in stats
    assert stats["global_noise_variance"] > 0


def test_perform_noise_analysis_spliced_noise():
    """Verify that splicing a patch with distinct noise variance elevates inconsistency metrics."""
    np.random.seed(42)
    # Background has low noise (sigma=2)
    base = np.full((160, 160, 3), 120, dtype=np.float32)
    bg_noise = np.random.normal(0, 2.0, (160, 160, 3))
    
    # Spliced region has extreme noise (sigma=35)
    spliced_noise = np.random.normal(0, 35.0, (60, 60, 3))
    
    composite = base + bg_noise
    composite[20:80, 20:80] = base[20:80, 20:80] + spliced_noise
    composite = np.clip(composite, 0, 255).astype(np.uint8)

    _, _, stats = perform_noise_analysis(composite, block_size=16)
    assert stats["std_local_variance"] > 0
    assert stats["inconsistent_blocks_count"] > 0
