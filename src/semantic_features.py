"""
semantic_features.py

Extract semantic features from OASIS images for valence/arousal prediction.

Two approaches are explored (run as separate experiments):
  1. Category labels  — Use the four OASIS-provided categories (Animals, Objects,
                        Scenery, People) as a one-hot encoded feature vector.
  2. Classifier-based — Replace hand-labeled categories with predictions from a
                        pretrained image classification model and measure the
                        difference in predictive performance.
"""

import numpy as np
from PIL import Image

# The four semantic categories defined by the OASIS dataset.
OASIS_CATEGORIES = ["Animals", "Objects", "Scenery", "People"]


def encode_category(category: str) -> np.ndarray:
    """
    One-hot encode an OASIS category label.

    Args:
        category: One of "Animals", "Objects", "Scenery", or "People".

    Returns:
        1-D binary array of length 4.
    """
    if category not in OASIS_CATEGORIES:
        raise ValueError(f"Unknown category '{category}'. Must be one of {OASIS_CATEGORIES}.")
    vec = np.zeros(len(OASIS_CATEGORIES), dtype=np.float32)
    vec[OASIS_CATEGORIES.index(category)] = 1.0
    return vec


def predict_category(image: Image.Image, model=None, transform=None) -> str:
    """
    Predict the semantic category of an image using a pretrained classifier
    (Experiment 2 — replaces hand-labeled OASIS categories).

    Args:
        image:     PIL Image (RGB).
        model:     Pretrained PyTorch classification model (loaded externally).
        transform: torchvision transform matching the model's expected input.

    Returns:
        Predicted category label string.
    """
    raise NotImplementedError


def extract_semantic_features(category: str = None, image: Image.Image = None,
                               model=None, transform=None) -> np.ndarray:
    """
    Return a semantic feature vector using either the provided OASIS category
    label (Experiment 1) or a classifier prediction (Experiment 2).

    Exactly one of `category` or `model` must be supplied.

    Args:
        category:  OASIS category string (Experiment 1).
        image:     PIL Image, required for Experiment 2.
        model:     Pretrained classifier, required for Experiment 2.
        transform: Model transform, required for Experiment 2.

    Returns:
        One-hot encoded 1-D array of length 4.
    """
    if category is not None:
        return encode_category(category)
    if model is not None:
        predicted = predict_category(image, model=model, transform=transform)
        return encode_category(predicted)
    raise ValueError("Provide either a category label or a pretrained model.")
