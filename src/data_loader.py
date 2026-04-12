"""
data_loader.py

Utilities for loading and preprocessing the OASIS image dataset.
"""

import os
import pandas as pd
from PIL import Image


OASIS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "oasis")


def load_oasis_metadata(csv_path: str) -> pd.DataFrame:
    """Load the OASIS ratings CSV and return a cleaned DataFrame."""
    df = pd.read_csv(csv_path)
    return df


def load_image(image_path: str) -> Image.Image:
    """Load a single image as a PIL Image."""
    return Image.open(image_path).convert("RGB")


def get_image_paths(image_dir: str) -> list[str]:
    """Return sorted list of image file paths from a directory."""
    supported = {".jpg", ".jpeg", ".png"}
    return sorted(
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in supported
    )
