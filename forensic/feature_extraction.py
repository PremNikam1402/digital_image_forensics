"""Unified Forensic Feature Extraction Pipeline.

Extracts a standardized 24-dimensional feature vector combining:
1. Error Level Analysis (ELA) statistics
2. Sensor noise residual and local variance inconsistency metrics
3. Histogram moments, Shannon entropy, and comb anomaly scores
4. Copy-Move feature matching and RANSAC inlier counts
5. Metadata authenticity indicators

Every feature is rigorously defined and documented for academic defense/viva.
"""

from __future__ import annotations

from typing import Dict, Any, List, Union, Tuple, Optional
from pathlib import Path
from PIL import Image
import numpy as np

from .ela import perform_ela
from .noise_analysis import perform_noise_analysis
from .histogram import perform_histogram_analysis
from .copy_move import detect_copy_move
from .metadata import extract_metadata


# List of all 24 feature names in standardized vector order
FEATURE_NAMES = [
    "ela_mean_error",            # 0: Mean ELA pixel difference (compression degradation)
    "ela_std_error",             # 1: Std dev of ELA error (compression spread)
    "ela_max_error",             # 2: Max ELA error (peak compression delta)
    "ela_p95_error",             # 3: 95th percentile ELA error (outlier error concentration)
    "ela_block_variance",        # 4: Variance across spatial blocks (compression uniformity)
    "ela_block_ratio",           # 5: Max-to-min block error ratio (spatial inconsistency)
    "noise_global_variance",     # 6: Variance of high-frequency noise residual
    "noise_global_std",          # 7: Std dev of high-frequency noise residual
    "noise_snr_db",              # 8: Estimated Signal-to-Noise Ratio (dB)
    "noise_kurtosis",            # 9: Kurtosis of residual (non-Gaussian sensor signature)
    "noise_skewness",            # 10: Skewness of residual distribution
    "noise_local_var_std",       # 11: Std dev of local block noise variances
    "noise_inconsistency_ratio", # 12: Ratio of blocks deviating >2.5 sigma from global noise
    "gray_mean",                 # 13: Mean image luminance
    "gray_std",                  # 14: Standard deviation of luminance
    "gray_skewness",             # 15: Skewness of luminance histogram
    "gray_kurtosis",             # 16: Kurtosis of luminance histogram
    "gray_entropy",              # 17: Shannon entropy of luminance (information richness)
    "rgb_entropy_mean",          # 18: Average Shannon entropy across R, G, B channels
    "hist_comb_ratio",           # 19: Comb gap ratio (contrast stretching / equalization)
    "cm_keypoints_count",        # 20: Total SIFT/ORB keypoints detected
    "cm_inliers_count",          # 21: RANSAC-verified duplicate keypoint pairs
    "meta_has_exif",             # 22: Binary flag (1 if EXIF metadata present, 0 if stripped)
    "meta_software_flag",        # 23: Binary flag (1 if editing software signature present)
]

FEATURE_DESCRIPTIONS = {
    "ela_mean_error": "Average absolute pixel difference after JPEG recompression at quality 90.",
    "ela_std_error": "Standard deviation of error level analysis; reflects error spread.",
    "ela_max_error": "Peak error level across all pixels; highlights extreme alterations.",
    "ela_p95_error": "95th percentile error; isolates spliced boundaries without extreme outliers.",
    "ela_block_variance": "Spatial variance across 16x16 ELA blocks; high values suggest mixed sources.",
    "ela_block_ratio": "Ratio between max and min block ELA means; measures regional discordance.",
    "noise_global_variance": "Variance of median-filter residual; measures overall sensor noise floor.",
    "noise_global_std": "Standard deviation of noise residual.",
    "noise_snr_db": "Signal-to-noise ratio in decibels calculated from luminance and noise power.",
    "noise_kurtosis": "Kurtosis of noise residual; deviates from ~0 under non-Gaussian tampering.",
    "noise_skewness": "Asymmetry of noise residual distribution.",
    "noise_local_var_std": "Standard deviation of local block noise variances across the image.",
    "noise_inconsistency_ratio": "Fraction of blocks exhibiting statistically abnormal local noise.",
    "gray_mean": "Global average luminance intensity [0-255].",
    "gray_std": "Global luminance standard deviation [0-255].",
    "gray_skewness": "Third standardized moment of luminance histogram.",
    "gray_kurtosis": "Fourth standardized moment of luminance histogram.",
    "gray_entropy": "Shannon entropy of grayscale histogram in bits; natural photos are ~7.0-7.8.",
    "rgb_entropy_mean": "Mean Shannon entropy across individual R, G, B channels.",
    "hist_comb_ratio": "Proportion of empty bins within active histogram range (comb effect indicator).",
    "cm_keypoints_count": "Total SIFT/ORB keypoints extracted from the image.",
    "cm_inliers_count": "Number of geometrically verified duplicate keypoint pairs (copy-move).",
    "meta_has_exif": "Indicator flag: 1 if image retains camera EXIF data, 0 if missing or stripped.",
    "meta_software_flag": "Indicator flag: 1 if image metadata references photo editing software.",
}


def extract_all_features(
    image: Union[Image.Image, np.ndarray],
    image_path: Optional[Union[str, Path]] = None,
    include_metadata: bool = True,
    copy_move_algo: str = "SIFT",
    ela_quality: int = 90
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """Extract full 24-dimensional forensic feature vector and component dictionaries.
    
    Parameters:
        image: PIL Image or RGB numpy array
        image_path: Optional path to evidence file on disk for metadata extraction
        include_metadata: Whether to extract metadata indicators (default True)
        copy_move_algo: 'SIFT' or 'ORB' (default 'SIFT')
        ela_quality: ELA quality factor (default 90)
        
    Returns:
        Tuple of:
          - feature_vector: 1D numpy array of shape (24,) with dtype float32
          - feature_dict: Dictionary mapping feature_name -> float value
          - full_forensic_results: Nested dictionary of full analytical outputs from each module
    """
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image).convert("RGB")
    else:
        pil_image = image.convert("RGB")

    # 1. Run Error Level Analysis
    ela_pil, ela_diff, ela_stats = perform_ela(pil_image, quality=ela_quality)

    # 2. Run Noise Analysis
    noise_vis, local_vars, noise_stats = perform_noise_analysis(pil_image)

    # 3. Run Histogram Analysis
    hist_stats = perform_histogram_analysis(pil_image)

    # 4. Run Copy-Move Forgery Detection
    cm_vis, cm_results = detect_copy_move(pil_image, algorithm=copy_move_algo)

    # 5. Extract Metadata
    if include_metadata and image_path is not None and Path(image_path).exists():
        meta_results = extract_metadata(image_path)
    else:
        meta_results = extract_metadata(pil_image)

    # Construct feature values mapping
    features: Dict[str, float] = {
        "ela_mean_error": float(ela_stats["mean_error"]),
        "ela_std_error": float(ela_stats["std_error"]),
        "ela_max_error": float(ela_stats["max_error"]),
        "ela_p95_error": float(ela_stats["percentile_95"]),
        "ela_block_variance": float(ela_stats["block_variance"]),
        "ela_block_ratio": float(ela_stats["block_discrepancy_ratio"]),
        
        "noise_global_variance": float(noise_stats["global_noise_variance"]),
        "noise_global_std": float(noise_stats["global_noise_std"]),
        "noise_snr_db": float(noise_stats["snr_db"]),
        "noise_kurtosis": float(noise_stats["residual_kurtosis"]),
        "noise_skewness": float(noise_stats["residual_skewness"]),
        "noise_local_var_std": float(noise_stats["std_local_variance"]),
        "noise_inconsistency_ratio": float(noise_stats["inconsistency_ratio"]),
        
        "gray_mean": float(hist_stats["gray_mean"]),
        "gray_std": float(hist_stats["gray_std"]),
        "gray_skewness": float(hist_stats["gray_skewness"]),
        "gray_kurtosis": float(hist_stats["gray_kurtosis"]),
        "gray_entropy": float(hist_stats["gray_entropy"]),
        "rgb_entropy_mean": float(np.mean([
            hist_stats["channel_entropies"]["red"],
            hist_stats["channel_entropies"]["green"],
            hist_stats["channel_entropies"]["blue"]
        ])),
        "hist_comb_ratio": float(hist_stats["comb_ratio"]),
        
        "cm_keypoints_count": float(cm_results["keypoints_count"]),
        "cm_inliers_count": float(cm_results["inliers_count"]),
        
        "meta_has_exif": 1.0 if meta_results["has_exif"] else 0.0,
        "meta_software_flag": 1.0 if meta_results["editing_software_detected"] else 0.0,
    }

    # Ensure vector matches canonical order and contains no NaNs or Infs
    vec = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    for idx, name in enumerate(FEATURE_NAMES):
        val = features.get(name, 0.0)
        if np.isnan(val) or np.isinf(val):
            val = 0.0
        vec[idx] = val

    full_results = {
        "ela": {
            "image": ela_pil,
            "stats": ela_stats,
        },
        "noise": {
            "vis_image": noise_vis,
            "local_vars": local_vars,
            "stats": noise_stats,
        },
        "histogram": hist_stats,
        "copy_move": {
            "annotated_image": cm_vis,
            "results": cm_results,
        },
        "metadata": meta_results,
    }

    return vec, features, full_results
