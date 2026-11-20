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


def load_pair(image_path, mask_path, scale=0.25):
    """Load one image as grayscale and its mask as a 0/1 array, both resized.

    The image is read with OpenCV's default flags, which apply the camera's
    rotation tag, so it lines up with the mask. The mask is stored as JPEG,
    so it is thresholded at 127 to remove the grey levels from compression.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise FileNotFoundError(f"Could not read {image_path} or {mask_path}")
    if image.shape != mask.shape:
        raise ValueError(f"Size mismatch: {image_path} {image.shape} vs {mask.shape}")

    if scale != 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        size = (image.shape[1], image.shape[0])
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

    mask = (mask > 127).astype(np.uint8)
    return image, mask

def split_dev_test(pairs, dev_ratio=0.7, seed=42):
    """Split the pairs into a development set and a test set.

    The order is shuffled with a fixed seed, so every member gets exactly the
    same split on every machine. All tuning uses the development set; the
    test set is used only once, at the end of the project.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    n_dev = round(len(pairs) * dev_ratio)

    dev = []
    for i in sorted(order[:n_dev]):
        dev.append(pairs[i])

    test = []
    for i in sorted(order[n_dev:]):
        test.append(pairs[i])

    return dev, test