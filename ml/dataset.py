"""Dataset Handling and Sample Data Generation Module.

Supports standard public image tampering datasets (CASIA, Columbia, CoMoFoD)
and generates verified sample datasets for instant demonstration and testing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import cv2

from forensic.feature_extraction import extract_all_features, FEATURE_NAMES


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def create_sample_dataset(
    output_dir: Path,
    num_samples: int = 24
) -> Tuple[List[Path], List[Path]]:
    """Create a verified out-of-the-box sample dataset with authentic and tampered images.
    
    Generates realistic authentic photos and tampered counterparts (copy-move,
    splicing, and contrast manipulations) to ensure immediate out-of-the-box
    model training and demonstration.
    """
    auth_dir = output_dir / "authentic"
    tamp_dir = output_dir / "tampered"
    auth_dir.mkdir(parents=True, exist_ok=True)
    tamp_dir.mkdir(parents=True, exist_ok=True)

    auth_files: List[Path] = []
    tamp_files: List[Path] = []

    np.random.seed(42)

    for i in range(num_samples):
        # Base realistic image dimensions
        w, h = 400, 300

        # --- Generate Authentic Image ---
        # Multi-frequency textured background with smooth lighting gradient
        x = np.linspace(0, 4 * np.pi, w)
        y = np.linspace(0, 3 * np.pi, h)
        xx, yy = np.meshgrid(x, y)
        
        # Natural scene synthesis (sky gradient + ground wave texture)
        base_r = np.clip(120 + 60 * np.sin(xx * 0.5 + i) + 30 * np.cos(yy * 0.3), 30, 240)
        base_g = np.clip(140 + 50 * np.cos(xx * 0.3) + 40 * np.sin(yy * 0.5 + i), 30, 240)
        base_b = np.clip(180 + 40 * np.sin(yy * 0.2) + 20 * np.cos(xx * 0.4), 30, 240)
        
        # Add realistic sensor Gaussian noise (uniform across whole image)
        sensor_noise = np.random.normal(0, 4.0, (h, w, 3))
        auth_array = np.clip(np.stack([base_r, base_g, base_b], axis=-1) + sensor_noise, 0, 255).astype(np.uint8)

        # Add natural geometric features (e.g. geometric textured shapes)
        auth_pil = Image.fromarray(auth_array)
        draw = ImageDraw.Draw(auth_pil)
        draw.ellipse([60 + (i*10)%50, 50, 160 + (i*10)%50, 150], fill=(220, 180, 70), outline=(180, 130, 40))
        draw.polygon([(220, 220), (320, 220), (270, 120)], fill=(80, 160, 90), outline=(50, 110, 60))

        auth_path = auth_dir / f"authentic_{i+1:03d}.jpg"
        auth_pil.save(auth_path, format="JPEG", quality=92)
        auth_files.append(auth_path)

        # --- Generate Tampered Counterpart ---
        tamp_pil = auth_pil.copy()

        if i % 3 == 0:
            # Type 1: Copy-Move Forgery (Cloning a region to another location)
            # Crop a distinctive textured circle and paste it at a distant location
            crop_box = (60 + (i*10)%50, 50, 160 + (i*10)%50, 150)
            cloned_region = tamp_pil.crop(crop_box)
            # Paste cloned region 180 pixels to the right
            tamp_pil.paste(cloned_region, (220, 50))
            
        elif i % 3 == 1:
            # Type 2: Image Splicing (Pasting from foreign source with distinct noise/color)
            foreign_patch = Image.new("RGB", (90, 80), color=(240, 60, 60))
            # High noise in foreign patch
            fp_arr = np.array(foreign_patch, dtype=np.float32) + np.random.normal(0, 18.0, (80, 90, 3))
            foreign_patch = Image.fromarray(np.clip(fp_arr, 0, 255).astype(np.uint8))
            tamp_pil.paste(foreign_patch, (150, 160))
            
        else:
            # Type 3: Non-linear Contrast Manipulation & Re-compression artifact
            # Apply severe contrast enhancement and double-compression
            enhancer = cv2.convertScaleAbs(np.array(tamp_pil), alpha=1.6, beta=-40)
            tamp_pil = Image.fromarray(enhancer)

        tamp_path = tamp_dir / f"tampered_{i+1:03d}.jpg"
        # Re-save with altered quality factor (creates distinct ELA and compression boundary)
        tamp_pil.save(tamp_path, format="JPEG", quality=75)
        tamp_files.append(tamp_path)

    return auth_files, tamp_files


def load_dataset_from_directory(
    root_dir: Path,
    max_samples_per_class: Optional[int] = None
) -> Tuple[List[Path], List[int]]:
    """Scan directory structure for authentic (label 0) and tampered (label 1) images.
    
    Supports:
    - root_dir/authentic vs root_dir/tampered (or forged / manipulated / spliced)
    - CASIA directory structures (CASIA1/Au, CASIA1/Sp; CASIA2/Au, CASIA2/Tp)
    """
    auth_paths: List[Path] = []
    tamp_paths: List[Path] = []

    auth_names = {"authentic", "au", "pristine", "original", "authentic_images"}
    tamp_names = {"tampered", "tp", "sp", "forged", "spliced", "copymove", "tampered_images"}

    for subdir in root_dir.iterdir():
        if not subdir.is_dir():
            continue
        lower_name = subdir.name.lower()
        if lower_name in auth_names:
            for ext in SUPPORTED_EXTENSIONS:
                auth_paths.extend(list(subdir.glob(f"*{ext}")))
                auth_paths.extend(list(subdir.glob(f"*{ext.upper()}")))
        elif lower_name in tamp_names:
            for ext in SUPPORTED_EXTENSIONS:
                tamp_paths.extend(list(subdir.glob(f"*{ext}")))
                tamp_paths.extend(list(subdir.glob(f"*{ext.upper()}")))

    # Deduplicate paths
    auth_paths = sorted(list(set(auth_paths)))
    tamp_paths = sorted(list(set(tamp_paths)))

    if max_samples_per_class is not None:
        auth_paths = auth_paths[:max_samples_per_class]
        tamp_paths = tamp_paths[:max_samples_per_class]

    image_paths = auth_paths + tamp_paths
    labels = [0] * len(auth_paths) + [1] * len(tamp_paths)

    return image_paths, labels


def extract_dataset_features(
    image_paths: List[Path],
    labels: List[int],
    cache_path: Optional[Path] = None,
    progress_callback: Optional[Any] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract 24-dimensional feature matrix X and label vector y with optional disk caching."""
    if cache_path is not None and cache_path.exists():
        data = np.load(cache_path)
        return data["X"], data["y"]

    X_list = []
    y_list = []

    total = len(image_paths)
    for idx, (path, label) in enumerate(zip(image_paths, labels)):
        try:
            with Image.open(path) as img:
                vec, _, _ = extract_all_features(
                    img,
                    image_path=path,
                    include_metadata=True,
                    copy_move_algo="SIFT",
                    ela_quality=90
                )
                X_list.append(vec)
                y_list.append(label)
        except Exception as e:
            print(f"Skipping corrupted sample {path}: {e}")
            continue

        if progress_callback:
            progress_callback(idx + 1, total)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=X, y=y)

    return X, y
