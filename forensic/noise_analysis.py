"""Noise Analysis and Residual Inconsistency Detection Module.

Isolates high-frequency sensor noise patterns using median residual filtering
and Laplacian operators. Analyzes local variance uniformity to detect spliced
or composited regions with mismatched noise profiles.
"""

from __future__ import annotations

from typing import Tuple, Dict, Any, Union
from PIL import Image
import numpy as np
import cv2
from scipy.stats import kurtosis, skew


def perform_noise_analysis(
    image: Union[Image.Image, np.ndarray],
    block_size: int = 16
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Perform image noise analysis and local variance estimation.
    
    Parameters:
        image: PIL Image or numpy array (RGB or Grayscale)
        block_size: Grid size for local variance calculation (default 16x16)
        
    Returns:
        Tuple of:
          - noise_map_vis: 8-bit normalized noise residual image for display
          - local_variance_map: Float array of block variances
          - stats: Dictionary of noise statistical metrics
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"), dtype=np.uint8)
    else:
        img_np = image.copy()
        if img_np.ndim == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    # Convert to grayscale float [0, 255]
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Method 1: Noise Residual Extraction using Median Filtering
    # The smooth image represents structural signal; the difference isolates sensor noise
    denoised = cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)
    noise_residual = gray - denoised

    # Method 2: High-pass Laplacian filter for frequency verification
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)

    # Global noise statistics
    global_noise_var = float(np.var(noise_residual))
    global_noise_std = float(np.std(noise_residual))
    
    signal_power = float(np.mean(gray ** 2))
    noise_power = max(global_noise_var, 1e-6)
    snr_db = float(10 * np.log10(signal_power / noise_power)) if signal_power > 0 else 0.0

    residual_flat = noise_residual.flatten()
    kurt = float(kurtosis(residual_flat))
    sk = float(skew(residual_flat))

    # Method 3: Local Block Variance Analysis
    h, w = gray.shape
    bh = max(8, block_size)
    bw = max(8, block_size)

    ny = h // bh
    nx = w // bw
    local_vars = np.zeros((ny, nx), dtype=np.float32)
    inconsistent_blocks = 0

    for i in range(ny):
        for j in range(nx):
            block = noise_residual[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
            var_val = float(np.var(block))
            local_vars[i, j] = var_val

    # Detect inconsistent regions where local noise variance deviates substantially
    mean_local_var = float(np.mean(local_vars)) if local_vars.size > 0 else 1.0
    std_local_var = float(np.std(local_vars)) if local_vars.size > 0 else 0.0
    
    threshold_high = mean_local_var + 2.5 * std_local_var
    threshold_low = max(0.0, mean_local_var - 2.0 * std_local_var)

    inconsistent_mask = (local_vars > threshold_high) | (local_vars < threshold_low)
    inconsistent_blocks = int(np.sum(inconsistent_mask))
    total_blocks = local_vars.size
    inconsistency_ratio = float(inconsistent_blocks / max(total_blocks, 1))

    # Normalized noise residual visualization (scaled centered at 128 for display)
    noise_vis = np.clip(128.0 + noise_residual * 4.0, 0, 255).astype(np.uint8)

    # Inconsistency judgment
    if inconsistency_ratio > 0.08 and std_local_var > 3.0:
        interpretation = (
            f"Significant noise inconsistency detected ({inconsistency_ratio * 100:.1f}% of blocks "
            "deviate from global noise distribution). Possible splicing from an image with a different ISO or camera sensor."
        )
        suspicious = True
    elif inconsistency_ratio > 0.03:
        interpretation = (
            "Mild local noise variance fluctuation observed. May be attributed to differing texture complexity or localized blurring."
        )
        suspicious = False
    else:
        interpretation = (
            "Noise distribution is uniform across all blocks. Sensor noise characteristics are consistent throughout the image."
        )
        suspicious = False

    stats = {
        "global_noise_variance": round(global_noise_var, 3),
        "global_noise_std": round(global_noise_std, 3),
        "snr_db": round(snr_db, 2),
        "residual_kurtosis": round(kurt, 3),
        "residual_skewness": round(sk, 3),
        "mean_local_variance": round(mean_local_var, 3),
        "std_local_variance": round(std_local_var, 3),
        "inconsistent_blocks_count": inconsistent_blocks,
        "total_blocks_count": total_blocks,
        "inconsistency_ratio": round(inconsistency_ratio, 4),
        "suspicious_noise": suspicious,
        "interpretation": interpretation,
    }

    return noise_vis, local_vars, stats
