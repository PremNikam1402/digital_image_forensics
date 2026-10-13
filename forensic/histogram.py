"""Histogram Analysis and Non-Linear Enhancement Detection Module.

Calculates grayscale and multi-channel RGB histograms, statistical moments
(Mean, Variance, Skewness, Kurtosis), Shannon Entropy, and detects non-linear
histogram anomalies such as comb-like spacing resulting from manipulation.
"""

from __future__ import annotations

from typing import Dict, Any, Union
from PIL import Image
import numpy as np
import cv2
from scipy.stats import kurtosis, skew


def compute_entropy(counts: np.ndarray) -> float:
    """Compute Shannon Entropy in bits for a given histogram frequency vector."""
    total = np.sum(counts)
    if total <= 0:
        return 0.0
    probs = counts / total
    nonzero_probs = probs[probs > 0]
    return float(-np.sum(nonzero_probs * np.log2(nonzero_probs)))


def detect_comb_effect(hist: np.ndarray) -> Tuple[bool, int, float]:
    """Detect comb-like gaps or periodic spikes caused by contrast adjustment/equalization.
    
    Returns:
        (is_comb_detected: bool, empty_bins_in_range: int, comb_ratio: float)
    """
    non_zero_indices = np.where(hist > 0)[0]
    if len(non_zero_indices) < 20:
        return False, 0, 0.0

    min_idx, max_idx = int(non_zero_indices[0]), int(non_zero_indices[-1])
    active_span = max_idx - min_idx + 1

    # Count empty bins between active boundaries
    active_slice = hist[min_idx:max_idx + 1]
    zero_gaps = int(np.sum(active_slice == 0))
    comb_ratio = float(zero_gaps / max(active_span, 1))

    # Severe comb effect occurs when > 15% of intermediate bins are completely zero
    is_comb = comb_ratio > 0.15 and active_span > 60

    return is_comb, zero_gaps, round(comb_ratio, 4)


def perform_histogram_analysis(
    image: Union[Image.Image, np.ndarray]
) -> Dict[str, Any]:
    """Perform comprehensive histogram and statistical analysis.
    
    Parameters:
        image: PIL Image or numpy array (RGB)
        
    Returns:
        Dictionary containing histograms, statistical moments, entropy,
        and non-linear tampering indicators.
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"), dtype=np.uint8)
    else:
        img_np = image.copy()
        if img_np.ndim == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Calculate 256-bin histograms
    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_r = cv2.calcHist([img_np], [0], None, [256], [0, 256]).flatten()
    hist_g = cv2.calcHist([img_np], [1], None, [256], [0, 256]).flatten()
    hist_b = cv2.calcHist([img_np], [2], None, [256], [0, 256]).flatten()

    # Grayscale statistics
    gray_flat = gray.flatten().astype(np.float64)
    gray_mean = float(np.mean(gray_flat))
    gray_std = float(np.std(gray_flat))
    if gray_std > 1e-4:
        gray_skew = float(skew(gray_flat))
        gray_kurt = float(kurtosis(gray_flat))
    else:
        gray_skew = 0.0
        gray_kurt = 0.0
    gray_entropy = compute_entropy(hist_gray)


    # Color channel statistics
    r_flat = img_np[:, :, 0].flatten().astype(np.float64)
    g_flat = img_np[:, :, 1].flatten().astype(np.float64)
    b_flat = img_np[:, :, 2].flatten().astype(np.float64)

    r_mean, r_std = float(np.mean(r_flat)), float(np.std(r_flat))
    g_mean, g_std = float(np.mean(g_flat)), float(np.std(g_flat))
    b_mean, b_std = float(np.mean(b_flat)), float(np.std(b_flat))

    r_entropy = compute_entropy(hist_r)
    g_entropy = compute_entropy(hist_g)
    b_entropy = compute_entropy(hist_b)

    # Detect comb-like artifacts (indicator of contrast manipulation or histogram stretching)
    is_comb, empty_bins, comb_ratio = detect_comb_effect(hist_gray)

    if is_comb:
        comb_summary = (
            f"Comb-like histogram pattern detected ({empty_bins} zero-frequency bins within active range). "
            "This pattern typically emerges after post-capture non-linear contrast stretching or histogram equalization."
        )
    else:
        comb_summary = "Histogram distribution is smooth and continuous without abnormal bin spacing."

    return {
        "hist_gray": hist_gray.tolist(),
        "hist_r": hist_r.tolist(),
        "hist_g": hist_g.tolist(),
        "hist_b": hist_b.tolist(),
        "gray_mean": round(gray_mean, 2),
        "gray_std": round(gray_std, 2),
        "gray_skewness": round(gray_skew, 3),
        "gray_kurtosis": round(gray_kurt, 3),
        "gray_entropy": round(gray_entropy, 3),
        "channel_means": {
            "red": round(r_mean, 2),
            "green": round(g_mean, 2),
            "blue": round(b_mean, 2),
        },
        "channel_stds": {
            "red": round(r_std, 2),
            "green": round(g_std, 2),
            "blue": round(b_std, 2),
        },
        "channel_entropies": {
            "red": round(r_entropy, 3),
            "green": round(g_entropy, 3),
            "blue": round(b_entropy, 3),
        },
        "comb_effect_detected": is_comb,
        "empty_bins_in_range": empty_bins,
        "comb_ratio": comb_ratio,
        "comb_summary": comb_summary,
    }
