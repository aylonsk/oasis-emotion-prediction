"""
color_features.py

Extract color composition features from images for valence/arousal prediction.

Loads a pre-trained logistic regression color classifier (built by
color_classifier.py) and uses it to classify every pixel in an image into
one of ~16 perceptual color bins.  The output feature vector per image is:
  - A percentage-composition vector (fraction of pixels in each bin).
  - A binary dominance mask (1 for each bin above a threshold, 0 otherwise).
"""

import os
import pickle

import numpy as np
from PIL import Image
import skimage.color

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "saved_models", "color_classifier.pkl"
)

N_DOMINANT = 5
DOMINANCE_THRESHOLD = 0.03  # a bin must cover >= 3 % of pixels to count as dominant


def load_color_classifier(path: str = MODEL_PATH):
    """
    Load the saved color classifier and its associated bin names.

    Returns:
        (model, bin_names, use_lab) tuple.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["bin_names"], payload["use_lab"]


def _pixels_to_lab(pixels_rgb: np.ndarray) -> np.ndarray:
    """Convert an (N, 3) uint8 RGB array to CIELAB features."""
    rgb_norm = (pixels_rgb / 255.0).reshape(1, -1, 3)
    lab = skimage.color.rgb2lab(rgb_norm)
    return lab.reshape(-1, 3)


def compute_bin_composition(image: Image.Image, model=None, bin_names=None,
                            use_lab: bool = True) -> np.ndarray:
    """
    Classify every pixel in *image* and return the fraction of pixels
    assigned to each color bin.

    Args:
        image:     PIL Image (RGB).
        model:     Trained sklearn classifier (loaded via load_color_classifier).
        bin_names: Ordered list of bin label strings.
        use_lab:   Whether the model expects CIELAB input.

    Returns:
        1-D array of length len(bin_names), sums to 1.0.
    """
    pixels = np.array(image).reshape(-1, 3)

    if use_lab:
        X = _pixels_to_lab(pixels)
    else:
        X = pixels.astype(float)

    predictions = model.predict(X)

    composition = np.zeros(len(bin_names), dtype=np.float64)
    label_to_idx = {name: i for i, name in enumerate(bin_names)}
    for label in predictions:
        idx = label_to_idx.get(label)
        if idx is not None:
            composition[idx] += 1

    total = composition.sum()
    if total > 0:
        composition /= total
    return composition


def extract_dominant_colors(composition: np.ndarray, n: int = N_DOMINANT,
                            threshold: float = DOMINANCE_THRESHOLD) -> np.ndarray:
    """
    Return the indices of the top-N most dominant bins from a composition vector.

    Bins below *threshold* are not considered dominant.  If fewer than *n* bins
    exceed the threshold, the remaining slots are filled with -1.

    Args:
        composition: 1-D composition vector from compute_bin_composition.
        n:           Max number of dominant bins to return.
        threshold:   Minimum fraction to qualify as dominant.

    Returns:
        1-D int array of length n (bin indices, or -1 for empty slots).
    """
    above = np.where(composition >= threshold)[0]
    sorted_above = above[np.argsort(composition[above])[::-1]]
    top = sorted_above[:n]

    result = np.full(n, -1, dtype=int)
    result[:len(top)] = top
    return result


def dominance_mask(composition: np.ndarray,
                   threshold: float = DOMINANCE_THRESHOLD) -> np.ndarray:
    """
    Binary mask over bins: 1 if the bin exceeds *threshold*, else 0.

    Returns:
        1-D float32 array, same length as composition.
    """
    return (composition >= threshold).astype(np.float32)


def extract_color_features(image: Image.Image, model=None, bin_names=None,
                           use_lab: bool = True) -> np.ndarray:
    """
    Build the full color feature vector for one image:
      [bin_composition | dominance_mask]

    Length = 2 * len(bin_names).

    Args:
        image:     PIL Image (RGB).
        model:     Trained color classifier.
        bin_names: Ordered list of bin label strings.
        use_lab:   Whether the model expects CIELAB input.

    Returns:
        1-D numpy feature vector.
    """
    if model is None or bin_names is None:
        model, bin_names, use_lab = load_color_classifier()

    comp = compute_bin_composition(image, model, bin_names, use_lab)
    mask = dominance_mask(comp)
    return np.concatenate([comp, mask])
