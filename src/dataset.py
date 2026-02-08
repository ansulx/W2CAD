from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from .data_io import clean_segment, iter_segments, list_signal_files, load_mat_signal
from .scalogram import compute_scalogram, scalogram_to_image


@dataclass(frozen=True)
class IndexItem:
    path: Path
    label: int
    offset: int


def build_index(
    root: Path,
    modality: str,
    window_size: int,
    stride: int,
    missing_value: float,
) -> List[IndexItem]:
    items: List[IndexItem] = []
    for sf in list_signal_files(root, modality):
        signal = load_mat_signal(sf.path)
        for start, _ in iter_segments(signal, window_size, stride, missing_value):
            items.append(IndexItem(sf.path, sf.label, start))
    return items


class ScalogramDataset(Dataset):
    def __init__(
        self,
        index: List[IndexItem],
        window_size: int,
        missing_value: float,
        downsample_factor: int,
        scales: np.ndarray,
        wavelet: str,
    ) -> None:
        self.index = index
        self.window_size = window_size
        self.missing_value = missing_value
        self.downsample_factor = max(1, int(downsample_factor))
        self.scales = scales
        self.wavelet = wavelet

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        item = self.index[idx]
        signal = load_mat_signal(item.path)
        segment = signal[item.offset : item.offset + self.window_size]
        segment = clean_segment(segment, self.missing_value)
        if self.downsample_factor > 1:
            segment = segment[:: self.downsample_factor]
        scal = compute_scalogram(segment, self.scales, self.wavelet)
        img = scalogram_to_image(scal)
        x = torch.from_numpy(img)
        y = torch.tensor(item.label, dtype=torch.long)
        return x, y
