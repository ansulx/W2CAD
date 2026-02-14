#!/usr/bin/env python3
"""
Train a 3D UNet with Monte Carlo Dropout on the ADAM dataset
(Aneurysm Detection And segMentation) for volumetric medical image
segmentation, then evaluate with MC Dropout uncertainty estimation.

Usage
-----
::

    python scripts/train_unet3d_mc.py --config configs/adam_unet3d.yaml

Pipeline
--------
1. Discover subject folders and split into train / val / test.
2. Train a 3D UNet with Dice + BCE combined loss, Adam optimiser, LR
   scheduling, and early stopping.
3. At test time, run MC Dropout with T forward passes per patch and
   aggregate predictions via sliding-window over the full volume.
4. Save segmentation maps, uncertainty maps, metrics, figures, and
   training history.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.unet3d import UNet3D, build_unet3d
from src.volume_dataset import (
    ADAMVolumeDataset,
    discover_subjects,
    load_nifti,
    normalize_intensity,
    save_nifti,
    split_subjects,
)
from src.mc_inference import (
    compute_uncertainty_metrics,
    mc_predict,
    mc_predict_volume,
)


# =========================================================================
# Loss functions
# =========================================================================

class DiceLoss(nn.Module):
    """Soft Dice loss for binary / multi-class segmentation."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : (B, C, D, H, W)
        targets : (B, 1, D, H, W) — values in {0, ..., C-1}
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets: (B, C, D, H, W)
        targets_long = targets.squeeze(1).long()  # (B, D, H, W)
        one_hot = F.one_hot(targets_long, num_classes)  # (B, D, H, W, C)
        one_hot = one_hot.permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)

        dims = (0, 2, 3, 4)  # reduce over batch and spatial dims
        intersection = (probs * one_hot).sum(dim=dims)
        union = probs.sum(dim=dims) + one_hot.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Average over classes (skip background if desired)
        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """Combined Dice + Binary Cross Entropy loss (common for medical segmentation)."""

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(logits, targets)
        # Cross entropy expects (B, C, ...) logits and (B, ...) long targets
        ce_loss = F.cross_entropy(logits, targets.squeeze(1).long())
        return self.dice_weight * dice_loss + self.bce_weight * ce_loss


# =========================================================================
# Metrics
# =========================================================================

def compute_dice(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> float:
    """Compute Dice coefficient for the foreground class (label 1)."""
    pred_bin = (pred == 1).float()
    target_bin = (target == 1).float()
    intersection = (pred_bin * target_bin).sum().item()
    return float((2.0 * intersection + smooth) / (pred_bin.sum().item() + target_bin.sum().item() + smooth))


# =========================================================================
# Training loop
# =========================================================================

def train_one_epoch(
    model: UNet3D,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Train for one epoch and return average metrics."""
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        preds = logits.argmax(dim=1)  # (B, D, H, W)
        dice = compute_dice(preds, masks.squeeze(1).long())
        running_loss += loss.item()
        running_dice += dice
        n_batches += 1

    return {
        "loss": running_loss / max(n_batches, 1),
        "dice": running_dice / max(n_batches, 1),
    }


@torch.no_grad()
def validate(
    model: UNet3D,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate (deterministic, no MC Dropout) and return average metrics."""
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        preds = logits.argmax(dim=1)
        dice = compute_dice(preds, masks.squeeze(1).long())
        running_loss += loss.item()
        running_dice += dice
        n_batches += 1

    return {
        "loss": running_loss / max(n_batches, 1),
        "dice": running_dice / max(n_batches, 1),
    }


# =========================================================================
# MC Dropout evaluation on full volumes
# =========================================================================

def evaluate_mc_dropout(
    model: UNet3D,
    subjects: List[str],
    data_root: Path,
    image_name: str,
    mask_name: Optional[str],
    normalize_method: str,
    patch_size: Tuple[int, int, int],
    T: int,
    overlap: float,
    device: torch.device,
    out_dir: Path,
) -> List[Dict[str, float]]:
    """Run MC Dropout inference on each test subject and save outputs."""
    results: List[Dict[str, float]] = []
    mc_dir = out_dir / "mc_predictions"
    mc_dir.mkdir(parents=True, exist_ok=True)

    for subj in subjects:
        print(f"  MC inference on subject {subj} (T={T}) ...")
        t0 = time.time()

        img_path = data_root / subj / image_name
        volume, affine = load_nifti(img_path)
        volume_norm = normalize_intensity(volume, method=normalize_method)

        mc_result = mc_predict_volume(
            model,
            volume_norm,
            patch_size=patch_size,
            T=T,
            overlap=overlap,
            device=device,
            batch_size=1,
        )

        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s")

        # Save segmentation and uncertainty maps
        save_nifti(
            mc_result["segmentation"].astype(np.float32),
            affine,
            mc_dir / f"{subj}_seg.nii.gz",
        )
        save_nifti(
            mc_result["predictive_entropy"],
            affine,
            mc_dir / f"{subj}_entropy.nii.gz",
        )
        save_nifti(
            mc_result["mutual_information"],
            affine,
            mc_dir / f"{subj}_mi.nii.gz",
        )

        # If we have ground truth, compute metrics
        if mask_name is not None:
            mask_path = data_root / subj / mask_name
            if mask_path.exists():
                gt, _ = load_nifti(mask_path)
                gt = (gt > 0).astype(np.int64)
                metrics = compute_uncertainty_metrics(
                    mc_result["segmentation"],
                    gt,
                    mc_result["predictive_entropy"],
                    mc_result["mutual_information"],
                )
                metrics["subject"] = subj  # type: ignore[assignment]
                metrics["inference_time_s"] = elapsed
                results.append(metrics)
                print(
                    f"    Dice={metrics['dice']:.4f}  "
                    f"entropy={metrics['mean_entropy']:.4f}  "
                    f"MI={metrics['mean_mi']:.4f}  "
                    f"unc_AUC={metrics['uncertainty_auc']:.4f}"
                )

    return results


# =========================================================================
# Figures
# =========================================================================

def save_training_curves(history: List[Dict], out_dir: Path) -> None:
    """Plot training/validation loss and Dice over epochs."""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]

    # Loss
    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    ax.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax.plot(epochs, [h["val_loss"] for h in history], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (Dice + CE)")
    ax.set_title("3D UNet Training Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "unet3d_loss.png")
    plt.close(fig)

    # Dice
    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    ax.plot(epochs, [h["train_dice"] for h in history], label="train")
    ax.plot(epochs, [h["val_dice"] for h in history], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice Coefficient")
    ax.set_title("3D UNet Dice Score")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "unet3d_dice.png")
    plt.close(fig)


def save_mc_summary_figure(mc_results: List[Dict], out_dir: Path) -> None:
    """Bar chart of per-subject Dice and uncertainty metrics."""
    if not mc_results:
        return
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    subjects = [r.get("subject", str(i)) for i, r in enumerate(mc_results)]
    dices = [r["dice"] for r in mc_results]
    entropies = [r["mean_entropy"] for r in mc_results]
    mis = [r["mean_mi"] for r in mc_results]

    x = np.arange(len(subjects))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(max(8, len(subjects) * 0.8), 5), dpi=160)
    bars1 = ax1.bar(x - width, dices, width, label="Dice", color="#2196F3")
    ax1.set_ylabel("Dice / Score")
    ax1.set_xlabel("Subject")
    ax1.set_xticks(x)
    ax1.set_xticklabels(subjects, rotation=45, ha="right")

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x, entropies, width, label="Pred. Entropy", color="#FF9800", alpha=0.7)
    bars3 = ax2.bar(x + width, mis, width, label="Mutual Info", color="#4CAF50", alpha=0.7)
    ax2.set_ylabel("Uncertainty")

    lines = [bars1, bars2, bars3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")
    ax1.set_title("MC Dropout: Dice & Uncertainty per Subject")
    ax1.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(fig_dir / "unet3d_mc_summary.png")
    plt.close(fig)

    # Scatter: Dice vs Entropy (should be negatively correlated)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    ax.scatter(entropies, dices, c="#2196F3", edgecolors="black", s=80, alpha=0.8)
    ax.set_xlabel("Mean Predictive Entropy")
    ax.set_ylabel("Dice Coefficient")
    ax.set_title("Dice vs Predictive Entropy (MC Dropout)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "unet3d_dice_vs_entropy.png")
    plt.close(fig)


# =========================================================================
# Configuration
# =========================================================================

def load_cfg(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train 3D UNet with MC Dropout on ADAM dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config (e.g. configs/adam_unet3d.yaml)",
    )
    args = parser.parse_args()
    cfg = load_cfg(Path(args.config))

    # ----- Paths ----------------------------------------------------------
    data_root = Path(cfg["data_root"]).expanduser().resolve()
    out_dir = Path(cfg["output_dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    image_name = str(cfg.get("image_name", "TOF.nii.gz"))
    mask_name = cfg.get("mask_name", "aneurysms.nii.gz")
    mask_name = str(mask_name) if mask_name else None

    # ----- Discover subjects & split --------------------------------------
    train_dir = data_root / cfg.get("train_folder", "training")
    test_dir = data_root / cfg.get("test_folder", "testing")

    train_subjects_all = discover_subjects(train_dir, image_name)
    test_subjects = discover_subjects(test_dir, image_name)

    val_frac = float(cfg.get("val_size", 0.15))
    seed = int(cfg.get("seed", 42))
    train_subjects, val_subjects, _ = split_subjects(
        train_subjects_all, val_frac=val_frac, test_frac=0.0, seed=seed
    )

    print(f"Subjects — train: {len(train_subjects)}, val: {len(val_subjects)}, test: {len(test_subjects)}")

    # ----- Tiny run -------------------------------------------------------
    tiny_run = bool(cfg.get("tiny_run", False))
    if tiny_run:
        max_subj = int(cfg.get("max_subjects_tiny", 2))
        train_subjects = train_subjects[:max_subj]
        val_subjects = val_subjects[:max(1, max_subj // 2)]
        test_subjects = test_subjects[:max(1, max_subj // 2)]
        print(f"[TINY RUN] train {len(train_subjects)}, val {len(val_subjects)}, test {len(test_subjects)}")

    # ----- Hyperparameters ------------------------------------------------
    patch_size = tuple(cfg.get("patch_size", [64, 64, 64]))
    patches_per_volume = int(cfg.get("patches_per_volume", 8))
    batch_size = int(cfg.get("batch_size", 2))
    lr = float(cfg.get("lr", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-5))
    epochs = int(cfg.get("epochs", 100))
    patience = int(cfg.get("patience", 15))
    min_delta = float(cfg.get("min_delta", 0.001))
    normalize_method = str(cfg.get("normalize", "zscore"))
    dropout_rate = float(cfg.get("dropout_rate", 0.1))
    mc_samples = int(cfg.get("mc_samples", 20))
    mc_overlap = float(cfg.get("mc_overlap", 0.25))
    model_preset = str(cfg.get("model_preset", "base"))
    in_channels = int(cfg.get("in_channels", 1))
    num_classes = int(cfg.get("num_classes", 2))
    fg_ratio = float(cfg.get("foreground_ratio", 0.5))
    augment = bool(cfg.get("augment", True))

    # ----- Device ---------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ----- Datasets -------------------------------------------------------
    if train_subjects:
        train_ds = ADAMVolumeDataset(
            root=train_dir,
            subjects=train_subjects,
            image_name=image_name,
            mask_name=mask_name,
            patch_size=patch_size,
            patches_per_volume=patches_per_volume,
            normalize=normalize_method,
            foreground_ratio=fg_ratio,
            augment=augment,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(cfg.get("num_workers", 0)),
            pin_memory=bool(cfg.get("pin_memory", False)),
        )
    else:
        train_loader = None
        print("WARNING: No training subjects found. Skipping training.")

    if val_subjects:
        val_ds = ADAMVolumeDataset(
            root=train_dir,
            subjects=val_subjects,
            image_name=image_name,
            mask_name=mask_name,
            patch_size=patch_size,
            patches_per_volume=max(1, patches_per_volume // 2),
            normalize=normalize_method,
            foreground_ratio=0.0,
            augment=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(cfg.get("num_workers", 0)),
            pin_memory=bool(cfg.get("pin_memory", False)),
        )
    else:
        val_loader = None

    # ----- Model ----------------------------------------------------------
    model = build_unet3d(
        in_channels=in_channels,
        num_classes=num_classes,
        preset=model_preset,
        dropout_rate=dropout_rate,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: 3D UNet ({model_preset}), {n_params:,} parameters, dropout={dropout_rate}")

    # ----- Optimiser & scheduler ------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler_cfg = cfg.get("scheduler", {})
    scheduler = None
    if scheduler_cfg.get("name") == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(scheduler_cfg.get("factor", 0.5)),
            patience=int(scheduler_cfg.get("patience", 5)),
            min_lr=float(scheduler_cfg.get("min_lr", 1e-7)),
        )
    elif scheduler_cfg.get("name") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=float(scheduler_cfg.get("min_lr", 1e-7)),
        )

    criterion = DiceBCELoss(
        dice_weight=float(cfg.get("dice_weight", 0.5)),
        bce_weight=float(cfg.get("bce_weight", 0.5)),
    )

    # ----- Training -------------------------------------------------------
    best_val_dice = -1.0
    bad_epochs = 0
    history: List[Dict] = []

    # Save run metadata
    meta = {
        "device": str(device),
        "torch_version": torch.__version__,
        "model_preset": model_preset,
        "n_params": n_params,
        "dropout_rate": dropout_rate,
        "mc_samples": mc_samples,
        "config": cfg,
    }
    with (out_dir / "run_unet3d.json").open("w") as f:
        json.dump(meta, f, indent=2)

    if train_loader is not None:
        print(f"\nTraining for up to {epochs} epochs (patience={patience}) ...")
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = (
                validate(model, val_loader, criterion, device) if val_loader else {"loss": 0.0, "dice": 0.0}
            )
            elapsed = time.time() - t0

            record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_dice": train_metrics["dice"],
                "val_loss": val_metrics["loss"],
                "val_dice": val_metrics["dice"],
                "lr": optimizer.param_groups[0]["lr"],
                "time_s": elapsed,
            }
            history.append(record)

            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f}  |  "
                f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}  ({elapsed:.1f}s)"
            )

            # Checkpoint
            if val_metrics["dice"] > best_val_dice + min_delta:
                best_val_dice = val_metrics["dice"]
                bad_epochs = 0
                torch.save(model.state_dict(), out_dir / "unet3d_best.pt")
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

            # Step scheduler
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics["dice"])
                else:
                    scheduler.step()

        # Save final model + history
        torch.save(model.state_dict(), out_dir / "unet3d_final.pt")
        with (out_dir / "history_unet3d.json").open("w") as f:
            json.dump(history, f, indent=2)
        save_training_curves(history, out_dir)
        print(f"Best val Dice: {best_val_dice:.4f}")

        # Load best checkpoint for MC evaluation
        best_path = out_dir / "unet3d_best.pt"
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
            print("Loaded best checkpoint for MC evaluation.")

    # ----- MC Dropout evaluation ------------------------------------------
    eval_subjects = test_subjects if test_subjects else val_subjects
    if eval_subjects:
        eval_root = test_dir if test_subjects else train_dir
        print(f"\nMC Dropout evaluation on {len(eval_subjects)} subjects (T={mc_samples}) ...")
        mc_results = evaluate_mc_dropout(
            model=model,
            subjects=eval_subjects,
            data_root=eval_root,
            image_name=image_name,
            mask_name=mask_name,
            normalize_method=normalize_method,
            patch_size=patch_size,
            T=mc_samples,
            overlap=mc_overlap,
            device=device,
            out_dir=out_dir,
        )

        if mc_results:
            # Save per-subject metrics
            with (out_dir / "mc_metrics.json").open("w") as f:
                json.dump(mc_results, f, indent=2)

            # Aggregate
            mean_dice = float(np.mean([r["dice"] for r in mc_results]))
            mean_ent = float(np.mean([r["mean_entropy"] for r in mc_results]))
            mean_mi = float(np.mean([r["mean_mi"] for r in mc_results]))
            mean_unc_auc = float(np.mean([r["uncertainty_auc"] for r in mc_results]))
            summary = {
                "mean_dice": mean_dice,
                "mean_entropy": mean_ent,
                "mean_mi": mean_mi,
                "mean_uncertainty_auc": mean_unc_auc,
                "num_subjects": len(mc_results),
                "mc_samples_T": mc_samples,
            }
            with (out_dir / "mc_summary.json").open("w") as f:
                json.dump(summary, f, indent=2)
            print(
                f"\nMC Summary — Dice={mean_dice:.4f}  Entropy={mean_ent:.4f}  "
                f"MI={mean_mi:.4f}  Unc-AUC={mean_unc_auc:.4f}"
            )
            save_mc_summary_figure(mc_results, out_dir)
    else:
        print("No test/val subjects available for MC evaluation.")

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
