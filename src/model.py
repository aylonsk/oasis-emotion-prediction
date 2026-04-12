"""
model.py

Model definition for predicting OASIS valence and arousal scores.

The primary model is a CNN (or MLP) that takes a combined feature vector of:
  - Color bin composition percentages  (from color_features.py)
  - Dominant color indicators          (from color_features.py)
  - Semantic category encoding         (from semantic_features.py)

A regression baseline (Ridge) is also provided for comparison.
Both models output two continuous values: valence and arousal.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_baseline_model(alpha: float = 1.0) -> Pipeline:
    """
    Regression baseline: feature scaling + Ridge regression.
    Predicts a single target (valence or arousal); call twice for both.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=alpha)),
    ])


def build_cnn_model(input_dim: int, hidden_dim: int = 128, output_dim: int = 2):
    """
    Build a CNN / MLP that maps the combined color + semantic feature vector
    to (valence, arousal) predictions.

    Args:
        input_dim:  Total number of input features (color bins + dominant colors + category).
        hidden_dim: Hidden layer width.
        output_dim: 2 (valence + arousal).

    Returns:
        torch.nn.Module
    """
    raise NotImplementedError
