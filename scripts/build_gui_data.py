"""
build_gui_data.py

Pre-compute the static assets used by docs/ (the GitHub Pages GUI):
  - thumbnails for every OASIS image (long edge 320 px, JPEG 75)
  - one predictions.json with true + Ridge-predicted valence/arousal,
    squared error, color composition, dominance mask, and category
    for every image with a CSV row.

Run after training the Experiment 1 Ridge models:

    python scripts/build_gui_data.py
"""

import json
import os
import pickle
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from data_loader import load_oasis_metadata, load_image, get_image_paths
from color_features import extract_color_features, load_color_classifier
from semantic_features import encode_category
from train import CATEGORY_CSV_TO_OASIS

CSV_PATH = os.path.join(ROOT, "data", "oasis", "OASIS.csv")
IMAGE_DIR = os.path.join(ROOT, "data", "oasis", "Images")
MODELS_DIR = os.path.join(ROOT, "models", "saved_models")
DOCS_DIR = os.path.join(ROOT, "docs")
THUMBS_DIR = os.path.join(DOCS_DIR, "thumbs")

THUMB_LONG_EDGE = 320
THUMB_QUALITY = 75

# Canonical hex per bin so the frontend can colour-tag the composition bars.
BIN_HEX = {
    "blue":   "#1f77b4",
    "brown":  "#8c564b",
    "gray":   "#7f7f7f",
    "green":  "#2ca02c",
    "orange": "#ff7f0e",
    "pink":   "#e377c2",
    "purple": "#9467bd",
    "red":    "#d62728",
    "tan":    "#d2b48c",
    "teal":   "#17becf",
    "yellow": "#ffd400",
}


def _load_ridge(name: str):
    with open(os.path.join(MODELS_DIR, name), "rb") as f:
        return pickle.load(f)


def _save_thumb(image: Image.Image, dest: str) -> None:
    w, h = image.size
    scale = THUMB_LONG_EDGE / max(w, h)
    if scale < 1.0:
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    image.save(dest, format="JPEG", quality=THUMB_QUALITY, optimize=True)


def main() -> None:
    os.makedirs(THUMBS_DIR, exist_ok=True)

    color_clf, bin_names, use_lab = load_color_classifier()
    valence_model = _load_ridge("valence_model_exp1.pkl")
    arousal_model = _load_ridge("arousal_model_exp1.pkl")

    metadata = load_oasis_metadata(CSV_PATH)
    theme_to_row = metadata.set_index("Theme").to_dict("index")
    image_paths = get_image_paths(IMAGE_DIR)

    items = []
    skipped = 0

    for image_path in image_paths:
        theme = os.path.splitext(os.path.basename(image_path))[0]
        row = theme_to_row.get(theme)
        if row is None:
            skipped += 1
            continue
        mapped = CATEGORY_CSV_TO_OASIS.get(row["Category"])
        if mapped is None:
            skipped += 1
            continue

        image = load_image(image_path)
        color_feats = extract_color_features(
            image, model=color_clf, bin_names=bin_names, use_lab=use_lab,
        )
        n = len(bin_names)
        composition = color_feats[:n]
        dominance = color_feats[n:]

        sem = encode_category(mapped)
        x = np.concatenate([color_feats, sem]).reshape(1, -1)
        pred_v = float(valence_model.predict(x)[0])
        pred_a = float(arousal_model.predict(x)[0])
        true_v = float(row["Valence_mean"])
        true_a = float(row["Arousal_mean"])

        thumb_name = f"{theme}.jpg"
        _save_thumb(image, os.path.join(THUMBS_DIR, thumb_name))

        items.append({
            "theme": theme,
            "category": row["Category"],
            "category_oasis": mapped,
            "thumb": f"thumbs/{thumb_name}",
            "true_v": round(true_v, 4),
            "true_a": round(true_a, 4),
            "pred_v": round(pred_v, 4),
            "pred_a": round(pred_a, 4),
            "se_v": round((pred_v - true_v) ** 2, 6),
            "se_a": round((pred_a - true_a) ** 2, 6),
            "composition": [round(float(v), 5) for v in composition],
            "dominance": [int(v) for v in dominance],
        })

    items.sort(key=lambda d: d["theme"])

    payload = {
        "bins": list(bin_names),
        "bin_hex": [BIN_HEX.get(b, "#999999") for b in bin_names],
        "categories": sorted({d["category"] for d in items}),
        "items": items,
    }

    out_path = os.path.join(DOCS_DIR, "predictions.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    print(f"Wrote {len(items)} items to {out_path}")
    print(f"Wrote {len(items)} thumbnails to {THUMBS_DIR}")
    if skipped:
        print(f"Skipped {skipped} image(s) without a matching CSV row or category.")


if __name__ == "__main__":
    main()
