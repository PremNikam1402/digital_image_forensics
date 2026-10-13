"""Image Preprocessing and Integrity Verification Module.

Handles file validation, cryptographic hashing (SHA-256), format checking,
and structural property extraction without modifying original evidence files.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import BinaryIO, Tuple, Union, Optional, Dict, Any

from PIL import Image, UnidentifiedImageError
import numpy as np


SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG", "TIFF", "BMP", "WEBP"}


def calculate_sha256(data_or_path: Union[str, Path, bytes, BinaryIO]) -> str:
    """Calculate cryptographic SHA-256 hash of raw image data.
    
    Preserves raw bytes strictly untouched to maintain forensic chain of custody.
    """
    hasher = hashlib.sha256()
    
    if isinstance(data_or_path, (str, Path)):
        file_path = Path(data_or_path)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
    elif isinstance(data_or_path, bytes):
        hasher.update(data_or_path)
    elif hasattr(data_or_path, "read"):
        # File-like object (e.g. BytesIO or UploadedFile)
        pos = data_or_path.tell() if hasattr(data_or_path, "tell") else None
        if pos is not None and pos != 0:
            data_or_path.seek(0)
        
        while chunk := data_or_path.read(65536):
            hasher.update(chunk)
            
        if pos is not None:
            data_or_path.seek(pos)
    else:
        raise ValueError(f"Unsupported data type for SHA-256 computation: {type(data_or_path)}")
        
    return hasher.hexdigest()


def validate_image(
    data_or_path: Union[str, Path, bytes, BinaryIO]
) -> Tuple[bool, str, Optional[Image.Image]]:
    """Validate image file format and structural integrity.
    
    Performs multi-step integrity checks:
    1. Checks if stream is readable and non-empty.
    2. Opens with Pillow and runs verify() to detect structural corruption.
    3. Re-opens image to verify raster decoding capability.
    4. Validates format against allowed forensic formats.
    
    Returns:
        Tuple of (is_valid: bool, status_message: str, image: Optional[Image.Image])
    """
    raw_bytes: bytes
    
    try:
        if isinstance(data_or_path, (str, Path)):
            p = Path(data_or_path)
            if not p.is_file():
                return False, f"File does not exist: {p}", None
            with open(p, "rb") as f:
                raw_bytes = f.read()
        elif isinstance(data_or_path, bytes):
            raw_bytes = data_or_path
        elif hasattr(data_or_path, "read"):
            pos = data_or_path.tell() if hasattr(data_or_path, "tell") else None
            if pos is not None:
                data_or_path.seek(0)
            raw_bytes = data_or_path.read()
            if pos is not None:
                data_or_path.seek(pos)
        else:
            return False, f"Unsupported data source type: {type(data_or_path)}", None

        if len(raw_bytes) == 0:
            return False, "File is empty (0 bytes).", None

        # Step 1: Open and verify image structure
        stream = io.BytesIO(raw_bytes)
        try:
            with Image.open(stream) as img_check:
                img_format = img_check.format
                if img_format and img_format.upper() not in SUPPORTED_FORMATS:
                    return (
                        False,
                        f"Unsupported format '{img_format}'. Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}",
                        None,
                    )
                img_check.verify()
        except (UnidentifiedImageError, SyntaxError, ValueError) as e:
            return False, f"Corrupted or invalid image data: {str(e)}", None

        # Step 2: Re-open image and load pixels into memory (verify() invalidates image object)
        stream_load = io.BytesIO(raw_bytes)
        loaded_image = Image.open(stream_load)
        loaded_image.load()

        if loaded_image.width <= 0 or loaded_image.height <= 0:
            return False, "Invalid image dimensions (zero or negative).", None

        return True, "Image successfully validated.", loaded_image

    except Exception as e:
        return False, f"Image validation failed: {str(e)}", None


def get_image_properties(
    image: Image.Image,
    raw_bytes: Optional[bytes] = None,
    filename: Optional[str] = None
) -> Dict[str, Any]:
    """Extract standard structural and forensic properties of an image."""
    width, height = image.size
    mode = image.mode
    img_format = image.format or "UNKNOWN"
    
    if raw_bytes is not None:
        file_size = len(raw_bytes)
        sha256_hash = calculate_sha256(raw_bytes)
    else:
        # Calculate from in-memory byte buffer
        buf = io.BytesIO()
        save_format = img_format if img_format != "UNKNOWN" else "PNG"
        try:
            image.save(buf, format=save_format)
            raw_bytes = buf.getvalue()
            file_size = len(raw_bytes)
            sha256_hash = calculate_sha256(raw_bytes)
        except Exception:
            file_size = 0
            sha256_hash = "N/A"

    channels = len(image.getbands())

    return {
        "filename": filename or "evidence_image",
        "format": img_format,
        "width": width,
        "height": height,
        "dimensions": f"{width} x {height}",
        "megapixels": round((width * height) / 1_000_000, 2),
        "color_mode": mode,
        "channels": channels,
        "file_size_bytes": file_size,
        "file_size_kb": round(file_size / 1024, 2),
        "file_size_mb": round(file_size / (1024 * 1024), 3),
        "sha256": sha256_hash,
    }


def to_rgb_array(image: Image.Image) -> np.ndarray:
    """Convert any PIL image (RGBA, L, CMYK, P) to standard 3-channel RGB numpy array (uint8)."""
    if image.mode != "RGB":
        rgb_image = image.convert("RGB")
    else:
        rgb_image = image
    return np.array(rgb_image, dtype=np.uint8)


def to_bgr_array(image: Image.Image) -> np.ndarray:
    """Convert PIL image to standard OpenCV BGR numpy array (uint8)."""
    rgb = to_rgb_array(image)
    return rgb[:, :, ::-1].copy()
