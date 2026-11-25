import numpy as np


def negative_transform_np(image: np.ndarray) -> np.ndarray:
    """Invert grayscale pixel intensities using NumPy."""
    return (255 - image).astype(np.uint8)


def log_transform_np(image: np.ndarray) -> np.ndarray:
    """Apply logarithmic intensity transformation using NumPy."""
    img_float = image.astype(np.float32)
    max_val = np.max(img_float)
    if max_val == 0:
        return image.copy()
    c = 255.0 / np.log(1.0 + max_val)
    transformed = c * np.log(1.0 + img_float)
    return np.clip(transformed, 0, 255).astype(np.uint8)


def gamma_correction_np(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Apply power-law (gamma) transformation using NumPy."""
    normalized = image.astype(np.float32) / 255.0
    corrected = np.power(normalized, gamma) * 255.0
    return np.clip(corrected, 0, 255).astype(np.uint8)