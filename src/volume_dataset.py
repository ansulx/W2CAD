"""
3D volume dataset for the ADAM challenge (Aneurysm Detection And segMentation).

The ADAM dataset provides TOF-MRA volumes as NIfTI files (.nii / .nii.gz) with
corresponding segmentation masks. This module provides:

1. ``ADAMVolumeDataset`` — a PyTorch Dataset that loads full volumes and
   optionally extracts random 3D patches during training.
2. ``PatchSampler3D`` — extracts random or grid patches from a volume.
3. Preprocessing utilities (resampling, intensity normalisation, padding).

Expected ADAM directory layout
------------------------------
::

    adam_root/
    ├── training/
    │   ├── 001/
    │   │   ├── TOF.nii.gz        # raw TOF-MRA volume
    │   │   └── aneurysms.nii.gz  # binary segmentation mask
    │   ├── 002/ ...
    └── testing/
        ├── 051/
        │   └── TOF.nii.gz
        ...

The filenames may vary; configure them via the YAML config or constructor
parameters ``image_name`` / ``mask_name``.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# NIfTI I/O  (nibabel-based, with graceful fallback)
# ---------------------------------------------------------------------------

def load_nifti(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI file and return ``(data, affine)``."""
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            "nibabel is required for NIfTI I/O. Install with: pip install nibabel"
        ) from exc
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    affine = np.asarray(img.affine, dtype=np.float64)
    return data, affine


def save_nifti(data: np.ndarray, affine: np.ndarray, path: Path) -> None:
    """Save a numpy array as a NIfTI file."""
    import nibabel as nib
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(path))


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def normalize_intensity(
    volume: np.ndarray,
    method: str = "zscore",
    clip_range: Tuple[float, float] = (0.5, 99.5),
) -> np.ndarray:
    """
    Normalise voxel intensities.

    Parameters
    ----------
    method : str
        ``"zscore"``  — zero-mean, unit-variance (percentile-clipped).
        ``"minmax"``  — scale to [0, 1] (percentile-clipped).
    clip_range : tuple
        Lower and upper percentiles for clipping before normalisation.
    """
    lo = float(np.percentile(volume, clip_range[0]))
    hi = float(np.percentile(volume, clip_range[1]))
    volume = np.clip(volume, lo, hi)

    if method == "zscore":
        mean = float(volume.mean())
        std = float(volume.std()) + 1e-8
        return (volume - mean) / std
    elif method == "minmax":
        vmin = float(volume.min())
        vmax = float(volume.max())
        if vmax - vmin < 1e-8:
            return np.zeros_like(volume)
        return (volume - vmin) / (vmax - vmin)
    else:
        raise ValueError(f"Unknown normalisation method: {method}")


def pad_to_divisible(
    volume: np.ndarray,
    divisor: int = 16,
    mode: str = "constant",
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Pad a 3D volume so each spatial dimension is divisible by ``divisor``.

    Returns ``(padded_volume, pad_widths)`` where *pad_widths* can be used to
    reverse the padding.
    """
    pad_widths: List[Tuple[int, int]] = []
    for s in volume.shape:
        remainder = s % divisor
        if remainder == 0:
            pad_widths.append((0, 0))
        else:
            total_pad = divisor - remainder
            pad_widths.append((total_pad // 2, total_pad - total_pad // 2))
    padded = np.pad(volume, pad_widths, mode=mode)
    return padded, pad_widths


def unpad(volume: np.ndarray, pad_widths: List[Tuple[int, int]]) -> np.ndarray:
    """Remove padding applied by :func:`pad_to_divisible`."""
    slices = []
    for (before, after), s in zip(pad_widths, volume.shape):
        end = s - after if after > 0 else s
        slices.append(slice(before, end))
    return volume[tuple(slices)]


# ---------------------------------------------------------------------------
# Patch sampling
# ---------------------------------------------------------------------------

class PatchSampler3D:
    """
    Extract fixed-size 3D patches from a volume.

    Supports both random sampling (training) and sliding-window grid sampling
    (inference / validation).
    """

    def __init__(self, patch_size: Tuple[int, int, int] = (64, 64, 64)) -> None:
        self.patch_size = patch_size

    def random_patch(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Extract a single random patch."""
        d, h, w = image.shape
        pd, ph, pw = self.patch_size
        sd = random.randint(0, max(0, d - pd))
        sh = random.randint(0, max(0, h - ph))
        sw = random.randint(0, max(0, w - pw))
        img_patch = image[sd : sd + pd, sh : sh + ph, sw : sw + pw]
        mask_patch = None
        if mask is not None:
            mask_patch = mask[sd : sd + pd, sh : sh + ph, sw : sw + pw]
        return img_patch, mask_patch

    def foreground_centered_patch(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract a patch centered on a random foreground voxel (if any exist).
        Falls back to a random patch if no foreground is present.
        """
        fg_coords = np.argwhere(mask > 0)
        if len(fg_coords) == 0:
            img_p, mask_p = self.random_patch(image, mask)
            assert mask_p is not None
            return img_p, mask_p

        idx = random.randint(0, len(fg_coords) - 1)
        center = fg_coords[idx]
        pd, ph, pw = self.patch_size

        starts = [
            max(0, min(c - ps // 2, s - ps))
            for c, ps, s in zip(center, self.patch_size, image.shape)
        ]
        sd, sh, sw = starts
        return (
            image[sd : sd + pd, sh : sh + ph, sw : sw + pw],
            mask[sd : sd + pd, sh : sh + ph, sw : sw + pw],
        )

    def grid_patches(
        self,
        image: np.ndarray,
        overlap: float = 0.25,
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int]]]:
        """
        Yield patches on a regular grid with specified overlap fraction.

        Returns ``[(patch, (start_d, start_h, start_w)), ...]``.
        """
        d, h, w = image.shape
        pd, ph, pw = self.patch_size
        stride_d = max(1, int(pd * (1 - overlap)))
        stride_h = max(1, int(ph * (1 - overlap)))
        stride_w = max(1, int(pw * (1 - overlap)))

        patches: List[Tuple[np.ndarray, Tuple[int, int, int]]] = []
        for sd in range(0, max(1, d - pd + 1), stride_d):
            for sh in range(0, max(1, h - ph + 1), stride_h):
                for sw in range(0, max(1, w - pw + 1), stride_w):
                    patch = image[sd : sd + pd, sh : sh + ph, sw : sw + pw]
                    patches.append((patch, (sd, sh, sw)))
        return patches


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class ADAMVolumeDataset(Dataset):
    """
    PyTorch Dataset for the ADAM challenge (or similar NIfTI-based medical
    imaging segmentation tasks).

    Parameters
    ----------
    root : Path
        Root directory containing subject folders.
    subjects : list of str
        Subject folder names to include (e.g. ``["001", "002", ...]``).
    image_name : str
        Filename of the image volume inside each subject folder.
    mask_name : str or None
        Filename of the segmentation mask (``None`` for test set).
    patch_size : tuple of int
        3D patch size ``(D, H, W)``.
    patches_per_volume : int
        Number of random patches to extract per volume per epoch.
    normalize : str
        Intensity normalisation method (``"zscore"`` or ``"minmax"``).
    foreground_ratio : float
        Fraction of patches that should be centered on foreground voxels
        (only effective when mask is available and has foreground).
    augment : bool
        Whether to apply random augmentations.
    """

    def __init__(
        self,
        root: Path,
        subjects: Sequence[str],
        image_name: str = "TOF.nii.gz",
        mask_name: Optional[str] = "aneurysms.nii.gz",
        patch_size: Tuple[int, int, int] = (64, 64, 64),
        patches_per_volume: int = 8,
        normalize: str = "zscore",
        foreground_ratio: float = 0.5,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.subjects = list(subjects)
        self.image_name = image_name
        self.mask_name = mask_name
        self.patches_per_volume = patches_per_volume
        self.normalize = normalize
        self.foreground_ratio = foreground_ratio
        self.augment = augment
        self.sampler = PatchSampler3D(patch_size)

        # Pre-validate that image files exist
        for subj in self.subjects:
            img_path = self.root / subj / self.image_name
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")

    def __len__(self) -> int:
        return len(self.subjects) * self.patches_per_volume

    def _load_volume(self, subj: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        img_path = self.root / subj / self.image_name
        image, _ = load_nifti(img_path)
        image = normalize_intensity(image, method=self.normalize)

        mask = None
        if self.mask_name is not None:
            mask_path = self.root / subj / self.mask_name
            if mask_path.exists():
                mask, _ = load_nifti(mask_path)
                mask = (mask > 0).astype(np.float32)

        return image, mask

    def _augment(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply random augmentations (flips, intensity jitter)."""
        # Random flips along each axis
        for axis in range(3):
            if random.random() < 0.5:
                image = np.flip(image, axis=axis).copy()
                if mask is not None:
                    mask = np.flip(mask, axis=axis).copy()

        # Random intensity jitter
        if random.random() < 0.3:
            scale = random.uniform(0.9, 1.1)
            shift = random.uniform(-0.1, 0.1)
            image = image * scale + shift

        # Random Gaussian noise
        if random.random() < 0.2:
            noise = np.random.normal(0, 0.05, size=image.shape).astype(np.float32)
            image = image + noise

        return image, mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        subj_idx = idx // self.patches_per_volume
        subj = self.subjects[subj_idx]
        image, mask = self._load_volume(subj)

        # Choose sampling strategy
        use_fg = (
            mask is not None
            and np.any(mask > 0)
            and random.random() < self.foreground_ratio
        )
        if use_fg:
            assert mask is not None
            img_patch, mask_patch = self.sampler.foreground_centered_patch(image, mask)
        else:
            img_patch, mask_patch = self.sampler.random_patch(image, mask)

        if self.augment:
            img_patch, mask_patch = self._augment(img_patch, mask_patch)

        # Ensure correct patch size via padding if volume is smaller than patch
        pd, ph, pw = self.sampler.patch_size
        if img_patch.shape != (pd, ph, pw):
            padded = np.zeros((pd, ph, pw), dtype=np.float32)
            sd, sh, sw = img_patch.shape
            padded[:sd, :sh, :sw] = img_patch
            img_patch = padded
            if mask_patch is not None:
                mask_padded = np.zeros((pd, ph, pw), dtype=np.float32)
                mask_padded[:sd, :sh, :sw] = mask_patch
                mask_patch = mask_padded

        # To tensors: add channel dim -> (1, D, H, W)
        x = torch.from_numpy(img_patch.copy()).unsqueeze(0).float()
        sample: Dict[str, torch.Tensor] = {"image": x, "subject": torch.tensor(subj_idx)}
        if mask_patch is not None:
            sample["mask"] = torch.from_numpy(mask_patch.copy()).unsqueeze(0).float()
        return sample


# ---------------------------------------------------------------------------
# Helpers for building train / val / test splits
# ---------------------------------------------------------------------------

def discover_subjects(root: Path, image_name: str = "TOF.nii.gz") -> List[str]:
    """Return sorted list of subject folder names that contain the image file."""
    subjects = []
    if not root.exists():
        return subjects
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / image_name).exists():
            subjects.append(d.name)
    return subjects


def split_subjects(
    subjects: List[str],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Split subject list into train / val / test by random shuffle."""
    rng = random.Random(seed)
    shuffled = list(subjects)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    test_subj = shuffled[:n_test]
    val_subj = shuffled[n_test : n_test + n_val]
    train_subj = shuffled[n_test + n_val :]
    return train_subj, val_subj, test_subj
