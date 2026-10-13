"""Digital Image Forensics and Tampering Detection - Forensic Module.
Contains image preprocessing, metadata analysis, ELA, noise analysis,
histogram statistics, copy-move detection, and unified feature extraction.
"""

from .preprocessing import validate_image, calculate_sha256, get_image_properties
from .metadata import extract_metadata
from .ela import perform_ela
from .noise_analysis import perform_noise_analysis
from .histogram import perform_histogram_analysis
from .copy_move import detect_copy_move
from .feature_extraction import extract_all_features, FEATURE_NAMES

__all__ = [
    "validate_image",
    "calculate_sha256",
    "get_image_properties",
    "extract_metadata",
    "perform_ela",
    "perform_noise_analysis",
    "perform_histogram_analysis",
    "detect_copy_move",
    "extract_all_features",
    "FEATURE_NAMES",
]
