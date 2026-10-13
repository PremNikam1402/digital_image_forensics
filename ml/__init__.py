"""Machine Learning Module for Image Forgery Detection.
Contains dataset handling, model training, evaluation, and inference.
"""

from .predict import ForensicPredictor
from .evaluate import evaluate_model
from .train import train_forensic_classifier

__all__ = [
    "ForensicPredictor",
    "evaluate_model",
    "train_forensic_classifier",
]
