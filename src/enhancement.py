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


def contrast_stretching_np(image: np.ndarray) -> np.ndarray:
    """Stretch image intensities to full [0, 255] range using NumPy."""
    img_float = image.astype(np.float32)
    min_val = np.min(img_float)
    max_val = np.max(img_float)
    if max_val - min_val == 0:
        return image.copy()
    stretched = (img_float - min_val) * (255.0 / (max_val - min_val))
    return np.clip(stretched, 0, 255).astype(np.uint8)


def histogram_equalization_np(image: np.ndarray) -> np.ndarray:
    """Compute global histogram equalization from scratch in NumPy."""
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    cdf = hist.cumsum()
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_min = cdf_masked.min()
    lut = (cdf_masked - cdf_min) * 255.0 / (image.size - cdf_min)
    lut = np.ma.filled(lut, 0).astype(np.uint8)
    return lut[image]