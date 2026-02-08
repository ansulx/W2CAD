from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import scipy.io as sio


@dataclass(frozen=True)
class SignalFile:
    path: Path
    label: int  # 1 = CAD, 0 = NONCAD


def list_signal_files(root: Path, modality: str) -> List[SignalFile]:
    modality = modality.lower()
    if modality not in {"abp", "ppg"}:
        raise ValueError("modality must be 'abp' or 'ppg'")

    cad_dir = root / ("ABPCAD" if modality == "abp" else "ppg_CAD")
    noncad_dir = root / ("ABPNONCAD" if modality == "abp" else "ppg_NONCAD")

    cad_files = [SignalFile(p, 1) for p in sorted(cad_dir.glob("*.mat"))]
    noncad_files = [SignalFile(p, 0) for p in sorted(noncad_dir.glob("*.mat"))]
    return cad_files + noncad_files


def load_mat_signal(path: Path) -> np.ndarray:
    data = sio.loadmat(path)
    if "val" not in data:
        raise KeyError(f"Missing 'val' key in {path}")
    val = data["val"]
    if val.ndim == 2 and val.shape[0] == 1:
        val = val[0]
    return val.astype(np.float32)


def iter_segments(
    signal: np.ndarray,
    window_size: int,
    stride: int,
    missing_value: float,
) -> Iterable[Tuple[int, np.ndarray]]:
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")

    n = signal.shape[0]
    for start in range(0, n - window_size + 1, stride):
        segment = signal[start : start + window_size]
        yield start, segment


def clean_segment(segment: np.ndarray, missing_value: float) -> np.ndarray:
    segment = segment.copy()
    mask = segment == missing_value
    if mask.any():
        valid = segment[~mask]
        fill = float(np.median(valid)) if valid.size else 0.0
        segment[mask] = fill
    return segment
