"""Tests for image preprocessing, validation, hashing, and property extraction."""

import io
from pathlib import Path
from PIL import Image
import numpy as np
import pytest

from forensic.preprocessing import (
    validate_image,
    calculate_sha256,
    get_image_properties,
    to_rgb_array,
    to_bgr_array,
)


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    """Generate in-memory valid JPEG image bytes."""
    img = Image.new("RGB", (100, 80), color=(120, 150, 180))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@pytest.fixture
def sample_png_bytes() -> bytes:
    """Generate in-memory valid PNG image bytes."""
    img = Image.new("RGBA", (120, 90), color=(100, 200, 100, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_calculate_sha256(sample_jpeg_bytes):
    """Test SHA-256 calculation matches hashlib on identical bytes."""
    import hashlib
    expected_hash = hashlib.sha256(sample_jpeg_bytes).hexdigest()
    calculated_hash = calculate_sha256(sample_jpeg_bytes)
    assert calculated_hash == expected_hash
    assert len(calculated_hash) == 64


def test_validate_valid_jpeg(sample_jpeg_bytes):
    """Verify that a valid JPEG passes validation."""
    is_valid, msg, img = validate_image(sample_jpeg_bytes)
    assert is_valid is True
    assert img is not None
    assert img.width == 100
    assert img.height == 80


def test_validate_valid_png(sample_png_bytes):
    """Verify that a valid PNG passes validation."""
    is_valid, msg, img = validate_image(sample_png_bytes)
    assert is_valid is True
    assert img is not None
    assert img.width == 120
    assert img.height == 90


def test_validate_corrupted_image():
    """Verify that corrupted data fails validation gracefully without crashing."""
    corrupted_bytes = b"\xFF\xD8\xFF\xE0\x00\x10JFIF" + b"CORRUPTED_RANDOM_DATA" * 50
    is_valid, msg, img = validate_image(corrupted_bytes)
    assert is_valid is False
    assert img is None
    assert "Corrupted" in msg or "failed" in msg


def test_validate_empty_file():
    """Verify that 0-byte input fails validation."""
    is_valid, msg, img = validate_image(b"")
    assert is_valid is False
    assert "empty" in msg.lower()


def test_get_image_properties(sample_jpeg_bytes):
    """Verify property extraction from a loaded image."""
    img = Image.open(io.BytesIO(sample_jpeg_bytes))
    props = get_image_properties(img, raw_bytes=sample_jpeg_bytes, filename="evidence.jpg")
    
    assert props["filename"] == "evidence.jpg"
    assert props["width"] == 100
    assert props["height"] == 80
    assert props["channels"] == 3
    assert props["file_size_bytes"] == len(sample_jpeg_bytes)
    assert len(props["sha256"]) == 64


def test_color_conversions(sample_png_bytes):
    """Verify conversions to standard RGB and BGR numpy arrays."""
    img = Image.open(io.BytesIO(sample_png_bytes))
    rgb_arr = to_rgb_array(img)
    assert rgb_arr.shape == (90, 120, 3)
    assert rgb_arr.dtype == np.uint8

    bgr_arr = to_bgr_array(img)
    assert bgr_arr.shape == (90, 120, 3)
    # Check that red and blue channels are swapped between RGB and BGR
    assert np.array_equal(rgb_arr[:, :, 0], bgr_arr[:, :, 2])
