"""Loading, splitting and controlled degradation of the dataset.

Owner: Mohamed Waly (Tech Lead + Data & Baselines)
"""

import os
from pathlib import Path

import cv2
import numpy as np

IMAGE_DIR = "rgb"
MASK_DIR = "BW"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_data_root(root=None):
    """Return the folder that contains the rgb/ and BW/ folders.

    Order: the root argument, then the CRACK_DATA_DIR environment
    variable, then the data/ folder of this repository.
    """
    if root is None:
        root = os.environ.get("CRACK_DATA_DIR")
    if root is None:
        root = Path(__file__).resolve().parents[1] / "data"
    root = Path(root)
    if not (root / IMAGE_DIR).is_dir() or not (root / MASK_DIR).is_dir():
        raise FileNotFoundError(f"Expected {IMAGE_DIR}/ and {MASK_DIR}/ inside {root}")
    return root


def _files_by_stem(folder):
    """Map each image file name without extension to its path."""
    files = {}
    for path in folder.iterdir():
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            files[path.stem] = path
    return files


def list_pairs(root=None):
    """Return a sorted list of (image_path, mask_path) pairs.

    Images and masks are matched by file name without extension, ignoring
    upper/lower case in the extension (e.g. rgb/357.JPG <-> BW/357.jpg).
    Raises an error if an image has no mask or a mask has no image.
    """
    root = get_data_root(root)
    images = _files_by_stem(root / IMAGE_DIR)
    masks = _files_by_stem(root / MASK_DIR)

    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        raise ValueError(
            f"Images without masks: {missing_masks}; masks without images: {missing_images}"
        )

    pairs = []
    for stem in sorted(images):
        pairs.append((images[stem], masks[stem]))
    return pairs