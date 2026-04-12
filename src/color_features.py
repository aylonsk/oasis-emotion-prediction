"""
color_features.py

Extract color composition features from images for valence/arousal prediction.

Approach:
  - Every pixel's hex code is mapped to one of 20-25 perceptual color bins.
  - The result is a percentage-composition vector (fraction of pixels per bin).
  - The top N most dominant color bins are also recorded as a separate feature.
"""

import numpy as np
from PIL import Image


# Placeholder: the 20-25 color bin definitions will be defined here.
# Each entry maps a human-readable color name to a representative RGB range or hex code.
COLOR_BINS: dict = {}  # e.g. {"red": ..., "sky_blue": ..., ...}

N_DOMINANT = 5  # Number of dominant colors to extract per image


def map_pixel_to_bin(pixel_rgb: tuple, color_bins: dict) -> str:
    """
    Map a single pixel's RGB value to the closest color bin label.

    Args:
        pixel_rgb:  (R, G, B) tuple for a single pixel.
        color_bins: Dict mapping bin labels to their representative color definitions.

    Returns:
        The label of the nearest color bin.
    """
    raise NotImplementedError


def compute_bin_composition(image: Image.Image, color_bins: dict) -> np.ndarray:
    """
    Perform a pixel-by-pixel analysis and return the percentage composition
    of each color bin across the entire image.

    Args:
        image:      PIL Image (RGB).
        color_bins: Dict mapping bin labels to color definitions.

    Returns:
        1-D array of length len(color_bins), where each value is the fraction
        of pixels assigned to that bin (sums to 1.0).
    """
    raise NotImplementedError


def extract_dominant_colors(image: Image.Image, n: int = N_DOMINANT) -> np.ndarray:
    """
    Return the indices (or labels) of the top-N most dominant color bins.

    Args:
        image: PIL Image (RGB).
        n:     Number of dominant colors to return.

    Returns:
        1-D array of length n containing the bin indices/labels, ordered by dominance.
    """
    raise NotImplementedError


def extract_color_features(image: Image.Image, color_bins: dict = COLOR_BINS) -> np.ndarray:
    """
    Combine bin composition percentages and dominant color flags into a single
    feature vector for downstream model training.

    Args:
        image:      PIL Image (RGB).
        color_bins: Dict mapping bin labels to color definitions.

    Returns:
        1-D numpy feature vector.
    """
    raise NotImplementedError
