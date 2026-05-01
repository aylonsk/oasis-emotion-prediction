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
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    class MLPRegressor(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, max(hidden_dim // 2, 8))
            self.fc_out = nn.Linear(max(hidden_dim // 2, 8), output_dim)
            self.dropout = nn.Dropout(p=0.2)
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x):
            x = x.view(x.size(0), -1).float()
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            out = self.fc_out(x)
            return out

    return MLPRegressor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
