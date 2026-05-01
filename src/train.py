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
from color_features import extract_color_features, COLOR_BINS, load_color_classifier
from semantic_features import extract_semantic_features, OASIS_CATEGORIES
from model import build_baseline_model

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved_models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "oasis")

N_FOLDS = 5


def build_feature_matrix(image_paths: list[str], metadata, color_model, bin_names,
                         use_lab: bool = True, filename_to_category: dict | None = None,
                         sem_model=None, sem_transform=None) -> np.ndarray:
    """
    For each image, extract color bin composition + dominant colors + semantic
    category and concatenate into a single feature vector.

    `color_model` and `bin_names` should be loaded once (via `load_color_classifier`).
    `filename_to_category` is an optional mapping from image basename -> OASIS category
    to avoid repeated metadata searching.
    """

    feature_matrix = []

    for image_path in image_paths:
        image = load_image(image_path)
        color_features = extract_color_features(image, model=color_model, bin_names=bin_names, use_lab=use_lab)

        base = os.path.basename(image_path)
        # If a semantic model is provided, use it (Experiment 2). Otherwise
        # fall back to mapped category (Experiment 1) or default.
        if sem_model is not None:
            semantic_features = extract_semantic_features(image=image, model=sem_model, transform=sem_transform)
        else:
            category = None
            if filename_to_category is not None:
                category = filename_to_category.get(base)
            if category is None:
                category = "Objects"
            semantic_features = extract_semantic_features(category=category)
        feature_vector = np.concatenate([color_features, semantic_features])
        feature_matrix.append(feature_vector)

    return np.vstack(feature_matrix)


def train(csv_path: str, image_dir: str, alpha: float = 1.0, sem_model_path: str | None = None, sem_transform=None):
    metadata = load_oasis_metadata(csv_path)
    image_paths = get_image_paths(image_dir)

    # Load color classifier once to avoid repeated disk I/O during extraction
    color_model, bin_names, use_lab = load_color_classifier()

    # Build a filename -> category mapping from metadata (cheap, once)
    filename_cols = [c for c in metadata.columns if any(k in c.lower() for k in ("file", "image", "img", "filename", "photo"))]
    category_cols = [c for c in metadata.columns if "category" in c.lower() or c.lower() in ("label", "labels")]

    filename_to_category = {}
    # Prefer explicit category column if available
    chosen_cat_col = category_cols[0] if category_cols else None
    if chosen_cat_col:
        if filename_cols:
            for _, row in metadata.iterrows():
                for fc in filename_cols:
                    try:
                        raw = str(row[fc])
                    except Exception:
                        continue
                    base = os.path.basename(raw)
                    val = row[chosen_cat_col]
                    if isinstance(val, str) and val in OASIS_CATEGORIES:
                        filename_to_category[base] = val
                        filename_to_category[raw] = val
        elif len(metadata) == len(image_paths):
            # fallback: assume rows align with image_paths order
            for i, p in enumerate(image_paths):
                try:
                    val = metadata.iloc[i][chosen_cat_col]
                except Exception:
                    val = None
                base = os.path.basename(p)
                if isinstance(val, str) and val in OASIS_CATEGORIES:
                    filename_to_category[base] = val

    # If no explicit category column, try to discover one
    if not filename_to_category:
        for c in metadata.columns:
            try:
                unique_vals = metadata[c].astype(str).unique()
            except Exception:
                continue
            if any(v in OASIS_CATEGORIES for v in unique_vals):
                # map rows
                if filename_cols:
                    for _, row in metadata.iterrows():
                        for fc in filename_cols:
                            try:
                                raw = str(row[fc])
                            except Exception:
                                continue
                            base = os.path.basename(raw)
                            val = row[c]
                            if isinstance(val, str) and val in OASIS_CATEGORIES:
                                filename_to_category[base] = val
                                filename_to_category[raw] = val
                elif len(metadata) == len(image_paths):
                    for i, p in enumerate(image_paths):
                        try:
                            val = metadata.iloc[i][c]
                        except Exception:
                            val = None
                        base = os.path.basename(p)
                        if isinstance(val, str) and val in OASIS_CATEGORIES:
                            filename_to_category[base] = val
                if filename_to_category:
                    break

    # Optionally load a semantic classifier (Experiment 2) if user supplied
    sem_model = None
    if sem_model_path is not None:
        try:
            import torch
            sem_model = torch.load(sem_model_path, map_location="cpu")
            try:
                sem_model.eval()
            except Exception:
                pass
        except Exception as e:
            raise RuntimeError(f"Failed to load semantic model from {sem_model_path}: {e}")
    X = None
    # Build X: if sem_model was provided externally (not via CLI), include it.
    X = build_feature_matrix(
        image_paths, metadata, color_model=color_model, bin_names=bin_names,
        use_lab=use_lab, filename_to_category=filename_to_category,
        sem_model=sem_model, sem_transform=sem_transform,
    )
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
    parser.add_argument("--sem-model", dest="sem_model", help="Path to a pretrained PyTorch semantic classifier (optional)")
    args = parser.parse_args()

    train(csv_path=args.csv, image_dir=args.images, alpha=args.alpha, sem_model_path=args.sem_model)
