"""
semantic_features.py

Extract semantic features from OASIS images for valence/arousal prediction.

Two approaches are explored (run as separate experiments):
  1. Category labels  — Use the four OASIS-provided categories (Animals, Objects,
                        Scenery, People) as a one-hot encoded feature vector.
  2. Classifier-based — Replace hand-labeled categories with predictions from a
                        pretrained image classification model and measure the
                        difference in predictive performance.
"""

import numpy as np
from PIL import Image

from typing import Optional

# Optional heavy imports deferred until needed
try:
    import torch
    import torchvision.transforms as T
except Exception:  # pragma: no cover - keep module import lightweight if torch missing
    torch = None
    T = None


# The four semantic categories defined by the OASIS dataset.
OASIS_CATEGORIES = ["Animals", "Objects", "Scenery", "People"]


def encode_category(category: str) -> np.ndarray:
    """One-hot encode an OASIS category label.

    Args:
        category: One of the strings in `OASIS_CATEGORIES`.

    Returns:
        1-D binary array of length 4 (dtype float32).
    """
    if category not in OASIS_CATEGORIES:
        raise ValueError(f"Unknown category '{category}'. Must be one of {OASIS_CATEGORIES}.")
    vec = np.zeros(len(OASIS_CATEGORIES), dtype=np.float32)
    vec[OASIS_CATEGORIES.index(category)] = 1.0
    return vec


def _default_transform() -> 'T.Compose':
    """Return a default torchvision transform for common pretrained models."""
    if T is None:
        raise RuntimeError("torchvision is required for default transforms")
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _map_name_to_oasis_category(name: str) -> Optional[str]:
    """Map a predicted class name (or label) to one of the OASIS categories.

    Uses simple keyword matching heuristics. Returns `None` if no confident
    mapping is found.
    """
    if not isinstance(name, str):
        return None
    n = name.lower()
    animal_kw = ["dog", "cat", "animal", "bird", "fish", "horse", "elephant",
                 "lion", "tiger", "bear", "monkey", "sheep", "cow", "deer"]
    people_kw = ["person", "man", "woman", "boy", "girl", "people", "human", "face", "head"]
    scenery_kw = ["mountain", "landscape", "valley", "forest", "sea", "ocean", "beach",
                  "river", "sky", "sunset", "sunrise", "lake", "island"]
    object_kw = ["car", "chair", "table", "cup", "bottle", "book", "phone", "computer",
                 "guitar", "keyboard", "clock", "lamp", "vehicle"]

    for kw in animal_kw:
        if kw in n:
            return "Animals"
    for kw in people_kw:
        if kw in n:
            return "People"
    for kw in scenery_kw:
        if kw in n:
            return "Scenery"
    for kw in object_kw:
        if kw in n:
            return "Objects"
    return None


def predict_category(image: Image.Image, model=None, transform=None) -> str:
    """Predict the semantic category of an image using a pretrained classifier.

    This function is intentionally flexible to support a variety of pretrained
    models:
      - If the model outputs 4 logits corresponding to the OASIS categories,
        the predicted index is mapped directly.
      - If the model exposes an index-to-class mapping (e.g. `idx_to_class`,
        `classes`, or invertible `class_to_idx`), the mapped label is used and
        heuristics map that label to an OASIS category.
      - As a fallback, keyword matching on the class name is used. If no
        mapping is found, `Objects` is returned as a conservative default.

    Args:
        image: PIL Image (will be converted to RGB if needed).
        model: A PyTorch model (inference-only). If `None`, raises.
        transform: torchvision transform to prepare the image. If `None`, a
                   sensible default is used.

    Returns:
        One of the strings in `OASIS_CATEGORIES`.
    """
    if model is None:
        raise ValueError("A pretrained `model` must be provided to predict_category")
    if torch is None:
        raise RuntimeError("PyTorch is required to use predict_category")

    if image.mode != "RGB":
        image = image.convert("RGB")

    if transform is None:
        transform = _default_transform()

    model_device = next(model.parameters(), None)
    device = model_device.device if model_device is not None else torch.device("cpu")

    model.eval()
    with torch.no_grad():
        inp = transform(image).unsqueeze(0).to(device)
        out = model(inp)
        # Support logits or (logits, aux) tuples
        if isinstance(out, (tuple, list)):
            out = out[0]
        # If model outputs probabilities/logits per class
        if out.ndim == 2:
            pred_idx = int(out.argmax(dim=1).item())
        else:
            # Unexpected shape, try squeezing
            pred_idx = int(torch.tensor(out).view(-1).argmax().item())

    # Try to resolve a human-readable label from the model if available
    label_name = None
    if hasattr(model, "idx_to_class"):
        try:
            label_name = model.idx_to_class[pred_idx]
        except Exception:
            label_name = None
    if label_name is None and hasattr(model, "classes"):
        try:
            label_name = model.classes[pred_idx]
        except Exception:
            label_name = None
    if label_name is None and hasattr(model, "class_to_idx"):
        try:
            inv = {v: k for k, v in model.class_to_idx.items()}
            label_name = inv.get(pred_idx)
        except Exception:
            label_name = None

    # If the model directly predicts four outputs, map indices to OASIS
    if out.shape[-1] == len(OASIS_CATEGORIES):
        return OASIS_CATEGORIES[pred_idx]

    # If we have a label name, attempt to map it heuristically
    if label_name is not None:
        mapped = _map_name_to_oasis_category(label_name)
        if mapped is not None:
            return mapped

    # As a conservative fallback attempt to map using the label from the
    # model's metadata (string of the index) or keyword heuristics on it.
    try:
        mapped = _map_name_to_oasis_category(str(label_name or pred_idx))
        if mapped is not None:
            return mapped
    except Exception:
        pass

    # Last resort: default to 'Objects'
    return "Objects"


def extract_semantic_features(category: str = None, image: Image.Image = None,
                               model=None, transform=None) -> np.ndarray:
    """Return a semantic feature vector using either the provided OASIS
    category label (Experiment 1) or a classifier prediction (Experiment 2).

    Exactly one of `category` or `model` must be supplied. When `model` is
    supplied, `image` must also be provided.

    Args:
        category: OASIS category string (Experiment 1).
        image: PIL Image, required for Experiment 2.
        model: Pretrained classifier, required for Experiment 2.
        transform: Model transform, required for Experiment 2.

    Returns:
        One-hot encoded 1-D array of length 4.
    """
    have_category = category is not None
    have_model = model is not None
    if have_category == have_model:
        raise ValueError("Provide exactly one of `category` or `model`.")
    if have_category:
        return encode_category(category)
    # model path
    if image is None:
        raise ValueError("`image` must be provided when using a pretrained `model`")
    predicted = predict_category(image, model=model, transform=transform)
    return encode_category(predicted)
