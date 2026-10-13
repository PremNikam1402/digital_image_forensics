"""Tests for SIFT/ORB Copy-Move forgery detection."""

from PIL import Image, ImageDraw
import numpy as np
import pytest

from forensic.copy_move import detect_copy_move


def test_copy_move_on_authentic_smooth_image():
    """Verify that an authentic image reports no significant copy-move evidence."""
    # Plain smooth gradient image with no duplicate objects
    x = np.linspace(0, 100, 200)
    y = np.linspace(0, 100, 200)
    xx, yy = np.meshgrid(x, y)
    arr = np.clip(xx + yy, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr).convert("RGB")

    annotated, results = detect_copy_move(img, algorithm="SIFT")
    
    assert results["evidence_found"] is False
    assert results["summary"] == "No significant copy-move evidence detected."
    assert annotated.shape == (200, 200, 3)


def test_copy_move_on_cloned_regions():
    """Verify detection when a textured region is copied and pasted elsewhere."""
    # Create image with high-contrast textured feature
    img = Image.new("RGB", (350, 250), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Draw a rich textured pattern at position 1 (50, 50)
    for r in range(10, 45, 5):
        draw.ellipse([50 - r, 50 - r, 50 + r, 50 + r], outline=(r * 5, 20, 200), width=2)
    draw.line([(20, 50), (80, 50)], fill=(0, 0, 0), width=3)
    draw.line([(50, 20), (50, 80)], fill=(0, 0, 0), width=3)

    # Crop the pattern and paste it at a distant position (220, 120)
    pattern = img.crop((10, 10, 90, 90))
    img.paste(pattern, (220, 120))

    annotated, results = detect_copy_move(img, algorithm="SIFT", min_spatial_distance=40.0)
    
    # Should detect keypoints and candidate matches between the two regions
    assert results["keypoints_count"] > 10
    assert results["candidate_matches_count"] > 0
