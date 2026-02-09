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
        augment: bool = False,
        time_shift_max: int = 0,
        noise_std: float = 0.0,
        normalize: str = "none",
    ) -> None:
        self.index = index
        self.window_size = window_size
        self.missing_value = missing_value
        self.downsample_factor = max(1, int(downsample_factor))
        self.scales = scales
        self.wavelet = wavelet
        self.augment = bool(augment)
        self.time_shift_max = max(0, int(time_shift_max))
        self.noise_std = float(noise_std)
        self.normalize = str(normalize).lower()
        self._imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        item = self.index[idx]
        signal = load_mat_signal(item.path)
        segment = signal[item.offset : item.offset + self.window_size]
        segment = clean_segment(segment, self.missing_value)
        if self.downsample_factor > 1:
            segment = segment[:: self.downsample_factor]
        if self.augment:
            if self.time_shift_max > 0:
                shift = np.random.randint(-self.time_shift_max, self.time_shift_max + 1)
                if shift != 0:
                    segment = np.roll(segment, shift)
            if self.noise_std > 0:
                scale = float(np.std(segment)) if segment.size else 0.0
                if scale > 0:
                    segment = segment + np.random.normal(0.0, self.noise_std * scale, size=segment.shape)
        scal = compute_scalogram(segment, self.scales, self.wavelet)
        img = scalogram_to_image(scal)
        x = torch.from_numpy(img)
        if self.normalize == "imagenet":
            x = (x - self._imagenet_mean) / self._imagenet_std
        y = torch.tensor(item.label, dtype=torch.long)
        return x, y
