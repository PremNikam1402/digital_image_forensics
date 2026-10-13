"""Error Level Analysis (ELA) Forensic Module.

Performs controlled JPEG re-compression, pixel-level difference analysis,
and statistical error distribution measurement to detect compression inconsistencies.
"""

from __future__ import annotations

import io
from typing import Tuple, Dict, Any, Union
from PIL import Image, ImageChops, ImageEnhance
import numpy as np


def perform_ela(
    image: Union[Image.Image, np.ndarray],
    quality: int = 90,
    scale: int = 15
) -> Tuple[Image.Image, np.ndarray, Dict[str, Any]]:
    """Compute Error Level Analysis on an image.
    
    Parameters:
        image: PIL Image or numpy array (RGB)
        quality: JPEG re-compression quality factor (1-100, standard forensic default 90-95)
        scale: Amplification multiplier for subtle compression artifacts (default 15)
        
    Returns:
        Tuple of:
          - ela_pil: Enhanced PIL Image visualization of ELA
          - diff_array: Raw unscaled float difference array (H, W, C)
          - stats: Dictionary of statistical ELA metrics
    """
    quality = max(1, min(100, int(quality)))
    
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            pil_orig = Image.fromarray(image).convert("RGB")
        elif image.shape[2] == 4:
            pil_orig = Image.fromarray(image).convert("RGB")
        else:
            pil_orig = Image.fromarray(image)
    else:
        pil_orig = image.convert("RGB")

    # Step 1: Re-save image to an in-memory byte buffer at the target JPEG quality
    buffer = io.BytesIO()
    pil_orig.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    pil_resaved = Image.open(buffer)

    # Step 2: Compute absolute pixel differences between original and recompressed
    orig_np = np.array(pil_orig, dtype=np.float32)
    resaved_np = np.array(pil_resaved, dtype=np.float32)
    
    raw_diff = np.abs(orig_np - resaved_np)  # shape (H, W, 3)
    gray_diff = np.mean(raw_diff, axis=2)     # shape (H, W)

    # Step 3: Compute enhanced ELA visualization
    # Scale difference by enhancement multiplier, clamped to 255
    scaled_diff = np.clip(raw_diff * scale, 0, 255).astype(np.uint8)
    ela_pil = Image.fromarray(scaled_diff)

    # Step 4: Compute rigorous statistical metrics
    mean_error = float(np.mean(gray_diff))
    std_error = float(np.std(gray_diff))
    max_error = float(np.max(gray_diff))
    p95_error = float(np.percentile(gray_diff, 95))
    p99_error = float(np.percentile(gray_diff, 99))
    
    # Block-level variance to detect local inconsistency (e.g. 16x16 blocks)
    h, w = gray_diff.shape
    bh, bw = max(8, h // 16), max(8, w // 16)
    grid_means = []
    for y in range(0, h - bh + 1, bh):
        for x in range(0, w - bw + 1, bw):
            block = gray_diff[y:y+bh, x:x+bw]
            grid_means.append(np.mean(block))
            
    block_var = float(np.var(grid_means)) if grid_means else 0.0
    block_mean_ratio = float(np.max(grid_means) / (np.min(grid_means) + 1e-5)) if grid_means else 1.0

    # Interpretation guideline
    if block_mean_ratio > 4.5 and p95_error > 8.0:
        interpretation = (
            "High ELA variance detected across regions. Different areas exhibit distinctly "
            "different compression error levels, which may indicate image splicing or multi-source composition."
        )
        suspicious = True
    elif block_mean_ratio > 2.5:
        interpretation = (
            "Moderate ELA variation observed. Variations may stem from natural high-frequency textures, "
            "edges, or mild localized editing."
        )
        suspicious = False
    else:
        interpretation = (
            "Uniform ELA response across the image. Compression degradation is consistent across regions, "
            "typical of single-source compression."
        )
        suspicious = False

    stats = {
        "quality_level": quality,
        "scale_factor": scale,
        "mean_error": round(mean_error, 3),
        "std_error": round(std_error, 3),
        "max_error": round(max_error, 3),
        "percentile_95": round(p95_error, 3),
        "percentile_99": round(p99_error, 3),
        "block_variance": round(block_var, 3),
        "block_discrepancy_ratio": round(block_mean_ratio, 2),
        "suspicious_ela": suspicious,
        "interpretation": interpretation,
        "disclaimer": (
            "Notice: ELA is an exploratory forensic indicator, not conclusive proof of tampering. "
            "High-contrast edges and textured objects naturally produce elevated error levels."
        )
    }

    return ela_pil, raw_diff, stats
