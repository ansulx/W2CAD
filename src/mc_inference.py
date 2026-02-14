"""
Monte Carlo Dropout inference utilities for 3D UNet.

Core idea
---------
At test time, keep dropout **active** and run T stochastic forward passes for
each input.  The T softmax outputs form an empirical posterior predictive
distribution, from which we extract:

- **Mean prediction** — averaged softmax probabilities (better calibrated than
  a single deterministic pass).
- **Predictive entropy** — Shannon entropy of the mean prediction (total
  uncertainty = aleatoric + epistemic).
- **Mutual information** — difference between predictive entropy and mean
  per-sample entropy (epistemic uncertainty only).
- **Variance map** — per-voxel variance across forward passes.

Usage
-----
::

    from src.unet3d import build_unet3d
    from src.mc_inference import mc_predict, mc_predict_volume

    model = build_unet3d(in_channels=1, num_classes=2)
    model.load_state_dict(torch.load("best.pt"))

    # Patch-level MC prediction
    result = mc_predict(model, patch_tensor, T=20, device="cuda")
    print(result["mean_prob"].shape)       # (C, D, H, W)
    print(result["predictive_entropy"].shape)  # (D, H, W)

    # Full-volume sliding-window MC prediction
    vol_result = mc_predict_volume(
        model, volume_np, patch_size=(64, 64, 64), T=20, device="cuda"
    )

References
----------
Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation.
Nair, T. et al. (2020). Exploring uncertainty measures in DL for medical imaging.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .unet3d import UNet3D
from .volume_dataset import PatchSampler3D, pad_to_divisible, unpad


# ---------------------------------------------------------------------------
# Core MC Dropout prediction (patch level)
# ---------------------------------------------------------------------------

@torch.no_grad()
def mc_predict(
    model: UNet3D,
    x: torch.Tensor,
    T: int = 20,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Run T stochastic forward passes with MC Dropout enabled.

    Parameters
    ----------
    model : UNet3D
        Model with dropout layers. ``enable_mc_dropout()`` will be called
        automatically.
    x : torch.Tensor
        Input tensor of shape ``(B, C, D, H, W)``.
    T : int
        Number of MC samples.
    device : torch.device or None
        Device to run on. Defaults to the device of the first model parameter.

    Returns
    -------
    dict with keys:
        ``mean_prob``            — (B, Classes, D, H, W) mean softmax
        ``predictive_entropy``   — (B, D, H, W) total uncertainty
        ``mutual_information``   — (B, D, H, W) epistemic uncertainty
        ``variance``             — (B, Classes, D, H, W) per-class variance
        ``samples``              — (T, B, Classes, D, H, W) all MC samples
    """
    if device is None:
        device = next(model.parameters()).device
    x = x.to(device)

    model.eval()
    model.enable_mc_dropout()

    samples: List[torch.Tensor] = []
    for _ in range(T):
        logits = model(x)
        probs = F.softmax(logits, dim=1)  # (B, C, D, H, W)
        samples.append(probs.cpu())

    model.disable_mc_dropout()

    # Stack: (T, B, C, D, H, W)
    stacked = torch.stack(samples, dim=0)

    # Mean prediction
    mean_prob = stacked.mean(dim=0)  # (B, C, D, H, W)

    # Variance
    variance = stacked.var(dim=0)  # (B, C, D, H, W)

    # Predictive entropy: H[E_q[p(y|x)]]
    eps = 1e-10
    predictive_entropy = -(mean_prob * torch.log(mean_prob + eps)).sum(dim=1)  # (B, D, H, W)

    # Per-sample entropy, then average
    per_sample_entropy = -(stacked * torch.log(stacked + eps)).sum(dim=2)  # (T, B, D, H, W)
    mean_entropy = per_sample_entropy.mean(dim=0)  # (B, D, H, W)

    # Mutual information (epistemic uncertainty) = predictive_entropy - mean_entropy
    mutual_info = predictive_entropy - mean_entropy  # (B, D, H, W)

    return {
        "mean_prob": mean_prob,
        "predictive_entropy": predictive_entropy,
        "mutual_information": mutual_info,
        "variance": variance,
        "samples": stacked,
    }


# ---------------------------------------------------------------------------
# Full-volume MC prediction (sliding window)
# ---------------------------------------------------------------------------

@torch.no_grad()
def mc_predict_volume(
    model: UNet3D,
    volume: np.ndarray,
    patch_size: Tuple[int, int, int] = (64, 64, 64),
    T: int = 20,
    overlap: float = 0.25,
    device: Optional[torch.device] = None,
    batch_size: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Run MC Dropout prediction on an entire 3D volume using sliding-window
    patch aggregation.

    Parameters
    ----------
    model : UNet3D
        Trained model.
    volume : np.ndarray
        3D volume of shape ``(D, H, W)`` (already normalised).
    patch_size : tuple
        Patch dimensions.
    T : int
        Number of MC samples per patch.
    overlap : float
        Overlap fraction between adjacent patches.
    device : torch.device or None
        Compute device.
    batch_size : int
        Number of patches to process in parallel.

    Returns
    -------
    dict with keys:
        ``mean_prob``            — (Classes, D, H, W)
        ``predictive_entropy``   — (D, H, W)
        ``mutual_information``   — (D, H, W)
        ``segmentation``         — (D, H, W) argmax of mean_prob
    """
    if device is None:
        device = next(model.parameters()).device

    # Pad volume so patches tile cleanly
    padded, pad_widths = pad_to_divisible(volume, divisor=max(patch_size))

    sampler = PatchSampler3D(patch_size)
    patches_and_coords = sampler.grid_patches(padded, overlap=overlap)

    num_classes = _infer_num_classes(model, patch_size, device)
    D, H, W = padded.shape

    # Accumulation arrays
    prob_sum = np.zeros((num_classes, D, H, W), dtype=np.float64)
    entropy_sum = np.zeros((D, H, W), dtype=np.float64)
    mi_sum = np.zeros((D, H, W), dtype=np.float64)
    count = np.zeros((D, H, W), dtype=np.float64)

    # Process patches in batches
    for batch_start in range(0, len(patches_and_coords), batch_size):
        batch = patches_and_coords[batch_start : batch_start + batch_size]
        patch_tensors = []
        coords_list = []
        for patch_np, coords in batch:
            # Ensure patch has correct size (pad if at boundary)
            pd, ph, pw = patch_size
            if patch_np.shape != (pd, ph, pw):
                padded_patch = np.zeros((pd, ph, pw), dtype=np.float32)
                sd, sh, sw = patch_np.shape
                padded_patch[:sd, :sh, :sw] = patch_np
                patch_np = padded_patch
            t = torch.from_numpy(patch_np).unsqueeze(0).unsqueeze(0).float()
            patch_tensors.append(t)
            coords_list.append(coords)

        x = torch.cat(patch_tensors, dim=0)  # (B, 1, pd, ph, pw)
        result = mc_predict(model, x, T=T, device=device)

        mean_prob_batch = result["mean_prob"].numpy()       # (B, C, pd, ph, pw)
        entropy_batch = result["predictive_entropy"].numpy() # (B, pd, ph, pw)
        mi_batch = result["mutual_information"].numpy()      # (B, pd, ph, pw)

        pd, ph, pw = patch_size
        for i, (sd, sh, sw) in enumerate(coords_list):
            # Actual extent may be smaller at volume boundary
            ed = min(sd + pd, D)
            eh = min(sh + ph, H)
            ew = min(sw + pw, W)
            ld, lh, lw = ed - sd, eh - sh, ew - sw

            prob_sum[:, sd:ed, sh:eh, sw:ew] += mean_prob_batch[i, :, :ld, :lh, :lw]
            entropy_sum[sd:ed, sh:eh, sw:ew] += entropy_batch[i, :ld, :lh, :lw]
            mi_sum[sd:ed, sh:eh, sw:ew] += mi_batch[i, :ld, :lh, :lw]
            count[sd:ed, sh:eh, sw:ew] += 1.0

    # Average overlapping regions
    count = np.maximum(count, 1.0)
    mean_prob = prob_sum / count[np.newaxis, ...]
    pred_entropy = entropy_sum / count
    pred_mi = mi_sum / count

    # Un-pad
    mean_prob_unpadded = np.stack(
        [unpad(mean_prob[c], pad_widths) for c in range(num_classes)], axis=0
    )
    pred_entropy = unpad(pred_entropy, pad_widths)
    pred_mi = unpad(pred_mi, pad_widths)

    segmentation = np.argmax(mean_prob_unpadded, axis=0)

    return {
        "mean_prob": mean_prob_unpadded.astype(np.float32),
        "predictive_entropy": pred_entropy.astype(np.float32),
        "mutual_information": pred_mi.astype(np.float32),
        "segmentation": segmentation.astype(np.int64),
    }


def _infer_num_classes(
    model: UNet3D,
    patch_size: Tuple[int, int, int],
    device: torch.device,
) -> int:
    """Run a tiny forward pass to discover the number of output classes."""
    model.eval()
    dummy = torch.zeros(1, 1, *patch_size, device=device)
    out = model(dummy)
    return out.shape[1]


# ---------------------------------------------------------------------------
# Uncertainty-based metrics
# ---------------------------------------------------------------------------

def compute_uncertainty_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    entropy_map: np.ndarray,
    mi_map: np.ndarray,
) -> Dict[str, float]:
    """
    Compute calibration and uncertainty quality metrics.

    Parameters
    ----------
    prediction : np.ndarray
        Predicted segmentation mask (D, H, W), integer labels.
    ground_truth : np.ndarray
        Ground truth segmentation mask (D, H, W), integer labels.
    entropy_map : np.ndarray
        Predictive entropy (D, H, W).
    mi_map : np.ndarray
        Mutual information / epistemic uncertainty (D, H, W).

    Returns
    -------
    dict with keys:
        ``dice``               — Dice coefficient (foreground class 1)
        ``mean_entropy``       — Average predictive entropy
        ``mean_mi``            — Average epistemic uncertainty
        ``entropy_correct``    — Mean entropy on correctly predicted voxels
        ``entropy_incorrect``  — Mean entropy on incorrectly predicted voxels
        ``uncertainty_auc``    — AUROC of entropy for detecting misclassifications
    """
    # Dice
    pred_fg = (prediction == 1).astype(np.float64)
    gt_fg = (ground_truth == 1).astype(np.float64)
    intersection = (pred_fg * gt_fg).sum()
    dice = float(2.0 * intersection / (pred_fg.sum() + gt_fg.sum() + 1e-8))

    correct = (prediction == ground_truth).ravel()
    ent = entropy_map.ravel()

    metrics: Dict[str, float] = {
        "dice": dice,
        "mean_entropy": float(ent.mean()),
        "mean_mi": float(mi_map.mean()),
        "entropy_correct": float(ent[correct].mean()) if correct.any() else 0.0,
        "entropy_incorrect": float(ent[~correct].mean()) if (~correct).any() else 0.0,
    }

    # Uncertainty AUROC: can entropy predict misclassifications?
    try:
        from sklearn.metrics import roc_auc_score
        errors = (~correct).astype(np.int64)
        if len(np.unique(errors)) > 1:
            metrics["uncertainty_auc"] = float(roc_auc_score(errors, ent))
        else:
            metrics["uncertainty_auc"] = 0.0
    except ImportError:
        metrics["uncertainty_auc"] = -1.0  # sklearn not available

    return metrics
