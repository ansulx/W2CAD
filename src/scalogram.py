from __future__ import annotations

from typing import Tuple

import numpy as np
import pywt


def compute_scalogram(
    segment: np.ndarray,
    scales: np.ndarray,
    wavelet: str = "morl",
) -> np.ndarray:
    # Continuous wavelet transform
    coef, _ = pywt.cwt(segment, scales, wavelet)
    power = np.abs(coef) ** 2
    power = np.log1p(power)
    return power.astype(np.float32)


def scalogram_to_image(scalogram: np.ndarray) -> np.ndarray:
    # Normalize to 0-1 and stack to 3 channels for ResNet
    min_v = float(scalogram.min())
    max_v = float(scalogram.max())
    if max_v - min_v < 1e-8:
        norm = np.zeros_like(scalogram, dtype=np.float32)
    else:
        norm = (scalogram - min_v) / (max_v - min_v)
    img = np.stack([norm, norm, norm], axis=0)
    return img
