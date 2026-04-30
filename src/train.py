"""
train.py

Training and evaluation pipeline for OASIS valence/arousal prediction.

Pipeline:
  1. Load OASIS images and ratings CSV.
  2. Extract color bin composition + dominant color features per image.
  3. Append semantic category features (Experiment 1: OASIS labels;
     Experiment 2: pretrained classifier predictions).
  4. Train model with K-fold cross validation.
  5. Report performance (log loss / MSE) and save the trained model.
"""

import os
import pickle
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from data_loader import load_oasis_metadata, load_image, get_image_paths
from color_features import extract_color_features, COLOR_BINS
from semantic_features import extract_semantic_features
from model import build_baseline_model

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved_models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "oasis")

N_FOLDS = 5


def build_feature_matrix(image_paths: list[str], metadata) -> np.ndarray:
    """
    For each image, extract color bin composition + dominant colors + semantic
    category and concatenate into a single feature vector.
    """

    for image_path in image_paths:
        image = load_image(image_path)
        color_features = extract_color_features(image)
        semantic_features = extract_semantic_features(image)
        feature_vector = np.concatenate([color_features, semantic_features])
        feature_matrix.append(feature_vector)

    return np.array(feature_matrix)


def train(csv_path: str, image_dir: str, alpha: float = 1.0):
    metadata = load_oasis_metadata(csv_path)
    image_paths = get_image_paths(image_dir)

    X = build_feature_matrix(image_paths, metadata)
    y_valence = metadata["Valence_mean"].values
    y_arousal = metadata["Arousal_mean"].values

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    valence_scores, arousal_scores = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        yv_train, yv_val = y_valence[train_idx], y_valence[val_idx]
        ya_train, ya_val = y_arousal[train_idx], y_arousal[val_idx]

        valence_model = build_baseline_model(alpha=alpha)
        valence_model.fit(X_train, yv_train)
        valence_scores.append(mean_squared_error(yv_val, valence_model.predict(X_val)))

        arousal_model = build_baseline_model(alpha=alpha)
        arousal_model.fit(X_train, ya_train)
        arousal_scores.append(mean_squared_error(ya_val, arousal_model.predict(X_val)))

        print(f"Fold {fold} — Valence MSE: {valence_scores[-1]:.4f}  "
              f"Arousal MSE: {arousal_scores[-1]:.4f}")

    print(f"\nMean Valence MSE: {np.mean(valence_scores):.4f}")
    print(f"Mean Arousal MSE: {np.mean(arousal_scores):.4f}")

    # Refit on all data and save
    final_valence = build_baseline_model(alpha=alpha)
    final_valence.fit(X, y_valence)
    final_arousal = build_baseline_model(alpha=alpha)
    final_arousal.fit(X, y_arousal)

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "valence_model.pkl"), "wb") as f:
        pickle.dump(final_valence, f)
    with open(os.path.join(MODELS_DIR, "arousal_model.pkl"), "wb") as f:
        pickle.dump(final_arousal, f)
    print("Models saved.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train OASIS emotion prediction models.")
    parser.add_argument("--csv", required=True, help="Path to OASIS ratings CSV.")
    parser.add_argument("--images", required=True, help="Directory of OASIS images.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization strength.")
    args = parser.parse_args()

    train(csv_path=args.csv, image_dir=args.images, alpha=args.alpha)
