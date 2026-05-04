"""
train.py

Training and evaluation pipeline for OASIS valence/arousal prediction.

Pipeline:
  1. Load OASIS images and ratings CSV.
  2. Extract color bin composition + dominant color features per image.
  3. Append semantic category features (Experiment 1: OASIS labels;
     Experiment 2: pretrained classifier predictions).
  4. Train model with K-fold cross validation.
  5. Report mean squared error per fold and save the trained models.
"""

import os
import pickle
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from data_loader import load_oasis_metadata, load_image, get_image_paths
from color_features import extract_color_features, load_color_classifier
from semantic_features import extract_semantic_features
from model import build_baseline_model

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved_models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "oasis")

N_FOLDS = 5

# MLP training defaults (used when --model mlp).
MLP_EPOCHS = 200
MLP_BATCH = 64
MLP_LR = 1e-3

# OASIS CSV uses singular category names; encode_category expects plural form.
CATEGORY_CSV_TO_OASIS = {
    "Animal": "Animals",
    "Object": "Objects",
    "Scene":  "Scenery",
    "Person": "People",
}


def _load_pretrained_classifier():
    """Load a torchvision pretrained model + transform for Experiment 2."""
    from torchvision import models
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.eval()
    transform = weights.transforms()
    return model, transform


def build_feature_matrix(
    image_paths: list[str],
    metadata,
    experiment: int = 1,
    sem_model=None,
    sem_transform=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each image with a matching CSV row, build [color_features | semantic_features]
    and the valence/arousal targets. Returns (X, y_valence, y_arousal), all row-aligned
    to the matched subset.

    experiment=1: use the OASIS Category label from the CSV.
    experiment=2: predict the category with a pretrained classifier.
    """
    color_clf, bin_names, use_lab = load_color_classifier()
    theme_to_row = metadata.set_index("Theme").to_dict("index")

    feature_matrix: list[np.ndarray] = []
    y_valence: list[float] = []
    y_arousal: list[float] = []
    skipped = 0

    for image_path in image_paths:
        theme = os.path.splitext(os.path.basename(image_path))[0]
        row = theme_to_row.get(theme)
        if row is None:
            skipped += 1
            continue

        image = load_image(image_path)
        color_features = extract_color_features(
            image, model=color_clf, bin_names=bin_names, use_lab=use_lab,
        )

        if experiment == 1:
            mapped = CATEGORY_CSV_TO_OASIS.get(row["Category"])
            if mapped is None:
                skipped += 1
                continue
            semantic_features = extract_semantic_features(category=mapped)
        elif experiment == 2:
            semantic_features = extract_semantic_features(
                image=image, model=sem_model, transform=sem_transform,
            )
        else:
            raise ValueError(f"Unknown experiment id: {experiment}")

        feature_matrix.append(np.concatenate([color_features, semantic_features]))
        y_valence.append(float(row["Valence_mean"]))
        y_arousal.append(float(row["Arousal_mean"]))

    if skipped:
        print(f"Skipped {skipped} image(s) without a matching CSV row.")

    return np.array(feature_matrix), np.array(y_valence), np.array(y_arousal)


def _print_fold_summary(valence_scores, arousal_scores):
    print(f"\nMean Valence MSE: {np.mean(valence_scores):.4f}")
    print(f"Mean Arousal MSE: {np.mean(arousal_scores):.4f}")


def _kfold_ridge(X, y_valence, y_arousal, alpha: float, experiment: int):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    valence_scores, arousal_scores = [], []

    for fold, (tr, va) in enumerate(kf.split(X), start=1):
        v = build_baseline_model(alpha=alpha); v.fit(X[tr], y_valence[tr])
        a = build_baseline_model(alpha=alpha); a.fit(X[tr], y_arousal[tr])
        valence_scores.append(mean_squared_error(y_valence[va], v.predict(X[va])))
        arousal_scores.append(mean_squared_error(y_arousal[va], a.predict(X[va])))
        print(f"Fold {fold} — Valence MSE: {valence_scores[-1]:.4f}  "
              f"Arousal MSE: {arousal_scores[-1]:.4f}")

    _print_fold_summary(valence_scores, arousal_scores)

    final_v = build_baseline_model(alpha=alpha); final_v.fit(X, y_valence)
    final_a = build_baseline_model(alpha=alpha); final_a.fit(X, y_arousal)

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, f"valence_model_exp{experiment}.pkl"), "wb") as f:
        pickle.dump(final_v, f)
    with open(os.path.join(MODELS_DIR, f"arousal_model_exp{experiment}.pkl"), "wb") as f:
        pickle.dump(final_a, f)
    print("Models saved.")


def _train_one_mlp(X_arr, Y_arr, input_dim, device, *,
                   epochs=MLP_EPOCHS, batch=MLP_BATCH, lr=MLP_LR):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from model import build_cnn_model

    net = build_cnn_model(input_dim=input_dim, output_dim=2).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    Xt = torch.from_numpy(X_arr).float().to(device)
    Yt = torch.from_numpy(Y_arr).float().to(device)
    loader = DataLoader(TensorDataset(Xt, Yt), batch_size=batch, shuffle=True)

    net.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(net(xb), yb).backward()
            opt.step()
    return net


def _kfold_mlp(X, y_valence, y_arousal, experiment: int):
    import torch
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"MLP device: {device}")

    Y = np.stack([y_valence, y_arousal], axis=1)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    valence_scores, arousal_scores = [], []

    for fold, (tr, va) in enumerate(kf.split(X), start=1):
        net = _train_one_mlp(X[tr], Y[tr], input_dim=X.shape[1], device=device)
        net.eval()
        with torch.no_grad():
            preds = net(torch.from_numpy(X[va]).float().to(device)).cpu().numpy()
        valence_scores.append(mean_squared_error(Y[va, 0], preds[:, 0]))
        arousal_scores.append(mean_squared_error(Y[va, 1], preds[:, 1]))
        print(f"Fold {fold} — Valence MSE: {valence_scores[-1]:.4f}  "
              f"Arousal MSE: {arousal_scores[-1]:.4f}")

    _print_fold_summary(valence_scores, arousal_scores)

    final = _train_one_mlp(X, Y, input_dim=X.shape[1], device=device)
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, f"mlp_model_exp{experiment}.pt")
    torch.save({"state_dict": final.state_dict(), "input_dim": X.shape[1]}, save_path)
    print("Models saved.")


def train(csv_path: str, image_dir: str, alpha: float = 1.0,
          experiment: int = 1, model_type: str = "ridge"):
    metadata = load_oasis_metadata(csv_path)
    image_paths = get_image_paths(image_dir)

    sem_model = sem_transform = None
    if experiment == 2:
        print("Loading pretrained classifier for Experiment 2 ...")
        sem_model, sem_transform = _load_pretrained_classifier()

    X, y_valence, y_arousal = build_feature_matrix(
        image_paths, metadata,
        experiment=experiment, sem_model=sem_model, sem_transform=sem_transform,
    )
    print(f"Feature matrix: X={X.shape}, y_valence={y_valence.shape}")

    if model_type == "ridge":
        _kfold_ridge(X, y_valence, y_arousal, alpha=alpha, experiment=experiment)
    elif model_type == "mlp":
        _kfold_mlp(X, y_valence, y_arousal, experiment=experiment)
    else:
        raise ValueError(f"Unknown --model: {model_type}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train OASIS emotion prediction models.")
    parser.add_argument("--csv", required=True, help="Path to OASIS ratings CSV.")
    parser.add_argument("--images", required=True, help="Directory of OASIS images.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization strength.")
    parser.add_argument("--experiment", type=int, default=1, choices=[1, 2],
                        help="1 = OASIS category labels; 2 = pretrained classifier predictions.")
    parser.add_argument("--model", default="ridge", choices=["ridge", "mlp"],
                        help="Regressor: Ridge (sklearn) or MLP (PyTorch).")
    args = parser.parse_args()

    train(csv_path=args.csv, image_dir=args.images, alpha=args.alpha,
          experiment=args.experiment, model_type=args.model)
