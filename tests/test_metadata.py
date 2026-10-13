"""Tests for metadata and EXIF forensics extraction."""

import io
from PIL import Image
import pytest

from forensic.metadata import extract_metadata


def test_missing_metadata_handled_gracefully():
    """Verify that an image without metadata returns 'Metadata unavailable' without crashing."""
    img = Image.new("RGB", (64, 64), color=(200, 100, 50))
    res = extract_metadata(img)
    
    assert res["has_exif"] is False
    assert "unavailable" in res["status"].lower()
    assert res["camera_make"] == "Not recorded"
    assert res["camera_model"] == "Not recorded"
    assert res["editing_software_detected"] is False


def test_metadata_extraction_with_exif_tags():
    """Verify extraction when EXIF tags are present."""
    img = Image.new("RGB", (100, 100), color=(50, 100, 150))
    
    exif = img.getexif()
    # 271: Make, 272: Model, 305: Software
    exif[271] = "Canon"
    exif[272] = "EOS 5D Mark IV"
    exif[305] = "Adobe Photoshop 2024"

    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    buf.seek(0)
    
    with Image.open(buf) as saved_img:
        res = extract_metadata(saved_img)

    assert res["has_exif"] is True
    assert res["camera_make"] == "Canon"
    assert res["camera_model"] == "EOS 5D Mark IV"
    assert "Adobe Photoshop" in res["software"]
    assert res["editing_software_detected"] is True
