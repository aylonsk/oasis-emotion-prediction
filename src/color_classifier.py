"""
color_classifier.py

Train a logistic regression on the XKCD color naming dataset, reduce to a
curated set of perceptually distinct color bins, and save the final classifier
to models/saved_models/ for use by color_features.py.

Usage:
    python color_classifier.py
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import skimage.color

XKCD_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "xkcd", "xkcd_teaching.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved_models")

BIN_NAMES = [
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "brown",
    "gray",
    "teal",
    "tan",
]


def load_xkcd(csv_path: str = XKCD_CSV) -> pd.DataFrame:
    """Load and clean the XKCD color naming CSV."""
    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)
    df.loc[df.term == "grey", "term"] = "gray"
    df = df[df["term"].apply(lambda s: len(s.split(" ", 1)) == 1)].copy()
    return df


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert an (N, 3) array of 0-255 RGB values to CIELAB."""
    rgb_norm = (rgb / 255.0).reshape(1, -1, 3)
    lab = skimage.color.rgb2lab(rgb_norm)
    return lab.reshape(-1, 3)


# ── Step 1: train on all single-word XKCD color terms ────────────────────────

def train_full_model(df: pd.DataFrame, top_n: int = 20, use_lab: bool = True):
    """
    Train a logistic regression on the top_n most frequent XKCD color terms.
    Returns the fitted model, accuracy scores, and the train/test split.
    """
    top_terms = df["term"].value_counts().head(top_n).index
    sub = df[df["term"].isin(top_terms)].copy()

    if use_lab:
        X = rgb_to_lab(sub[["r", "g", "b"]].to_numpy(dtype=float))
    else:
        X = sub[["r", "g", "b"]].to_numpy(dtype=float)
    y = sub["term"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    model = LogisticRegression(max_iter=1000, tol=0.01, solver="saga")
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    f1 = f1_score(y_test, model.predict(X_test), average=None)

    print(f"=== Full model ({top_n} classes, {'CIELAB' if use_lab else 'RGB'}) ===")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")
    print("Per-class F1:")
    for cls, score in zip(model.classes_, f1):
        print(f"  {cls:>12s}: {score:.4f}")

    return model


# ── Step 2: retrain on the reduced, distinct bin set ──────────────────────────

def train_bin_model(df: pd.DataFrame, use_lab: bool = True):
    """
    Retrain the logistic regression using only the curated BIN_NAMES subset.
    Returns the fitted model.
    """
    sub = df[df["term"].isin(BIN_NAMES)].copy()
    missing = set(BIN_NAMES) - set(sub["term"].unique())
    if missing:
        print(f"Warning: these bins have no XKCD data and will be dropped: {missing}")

    if use_lab:
        X = rgb_to_lab(sub[["r", "g", "b"]].to_numpy(dtype=float))
    else:
        X = sub[["r", "g", "b"]].to_numpy(dtype=float)
    y = sub["term"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    model = LogisticRegression(max_iter=1000, tol=0.01, solver="saga")
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    print(f"\n=== Bin model ({len(model.classes_)} classes, {'CIELAB' if use_lab else 'RGB'}) ===")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")
    print("\nClassification report (test set):")
    print(classification_report(y_test, model.predict(X_test)))

    return model


# ── Step 3: save ──────────────────────────────────────────────────────────────

def save_model(model, bin_names: list[str], out_dir: str = MODELS_DIR):
    """Persist the trained classifier and bin name list to disk."""
    os.makedirs(out_dir, exist_ok=True)
    payload = {"model": model, "bin_names": bin_names, "use_lab": True}
    path = os.path.join(out_dir, "color_classifier.pkl")
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nSaved to {path}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    df = load_xkcd()
    print(f"Loaded {len(df)} XKCD rows ({df['term'].nunique()} unique terms)\n")

    train_full_model(df, top_n=20, use_lab=True)
    bin_model = train_bin_model(df, use_lab=True)

    actual_bins = sorted(set(BIN_NAMES) & set(bin_model.classes_))
    save_model(bin_model, actual_bins)


if __name__ == "__main__":
    main()
