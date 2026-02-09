from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import json
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import ScalogramDataset
from src.index_io import load_index_csv


def build_model(num_classes: int = 2) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_cfg(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def make_loader(
    cfg: dict,
    index_csv: Path,
    shuffle: bool,
    is_train: bool,
    max_samples: int | None = None,
) -> DataLoader:
    scales = np.array(cfg["scales"], dtype=np.float32)
    seed = int(cfg.get("seed", 42))
    index = load_index_csv(index_csv, max_samples=max_samples, seed=seed)
    augment = bool(cfg.get("augment", False)) and is_train
    time_shift_max = int(cfg.get("time_shift_max", 0))
    noise_std = float(cfg.get("noise_std", 0.0))
    normalize = str(cfg.get("normalize", "none"))
    dataset = ScalogramDataset(
        index=index,
        window_size=int(cfg["window_size"]),
        missing_value=float(cfg["missing_value"]),
        downsample_factor=int(cfg.get("downsample_factor", 1)),
        scales=scales,
        wavelet=str(cfg["wavelet"]),
        augment=augment,
        time_shift_max=time_shift_max,
        noise_std=noise_std,
        normalize=normalize,
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=shuffle,
        num_workers=int(cfg["num_workers"]),
        pin_memory=bool(cfg.get("pin_memory", False)),
    )


def forward_batch(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    # Resize to 224x224 for ResNet
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    return model(x)


def sensitivity_at_specificity(labels: np.ndarray, probs: np.ndarray, target_spec: float) -> float:
    fpr, tpr, _ = roc_curve(labels, probs)
    spec = 1.0 - fpr
    mask = spec >= target_spec
    if not np.any(mask):
        return 0.0
    return float(np.max(tpr[mask]))


def evaluate_probs(labels: np.ndarray, probs: np.ndarray) -> dict:
    preds = (probs >= 0.5).astype(np.int64)
    acc = float(accuracy_score(labels, preds))
    auc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0
    ap = float(average_precision_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0
    s95 = sensitivity_at_specificity(labels, probs, target_spec=0.95) if len(np.unique(labels)) > 1 else 0.0
    return {"acc": acc, "auc": auc, "ap": ap, "s95": s95}


def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = forward_batch(model, x)
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.detach().cpu().numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return probs, labels


def save_history(out_dir: Path, modality: str, history: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"history_{modality}.json").open("w") as f:
        json.dump(history, f, indent=2)


def save_run_metadata(out_dir: Path, modality: str, cfg: dict, device: torch.device) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "device": str(device),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "config": cfg,
    }
    with (out_dir / f"run_{modality}.json").open("w") as f:
        json.dump(meta, f, indent=2)


def _save_placeholder_plot(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
    ax.set_title(title)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_figures(
    out_dir: Path,
    modality: str,
    history: list[dict],
    labels: np.ndarray,
    probs: np.ndarray,
) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Training curves
    epochs = [h["epoch"] for h in history]
    for key, title in [
        ("loss", "Loss"),
        ("auc", "AUROC"),
        ("acc", "Accuracy"),
        ("ap", "AUPRC"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
        ax.plot(epochs, [h[f"train_{key}"] for h in history], label="train")
        ax.plot(epochs, [h[f"val_{key}"] for h in history], label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(f"{modality.upper()} {title}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f"{modality}_{key}.png")
        plt.close(fig)

    # Confusion matrix
    preds = (probs >= 0.5).astype(np.int64)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4), dpi=160)
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{modality.upper()} Confusion Matrix (thr=0.5)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{modality}_cm.png")
    plt.close(fig)

    # ROC and PR curves (skip if single class)
    if len(np.unique(labels)) > 1:
        fpr, tpr, _ = roc_curve(labels, probs)
        fig, ax = plt.subplots(figsize=(5.5, 4), dpi=160)
        ax.plot(fpr, tpr, label="ROC")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{modality.upper()} ROC Curve")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{modality}_roc.png")
        plt.close(fig)

        precision, recall, _ = precision_recall_curve(labels, probs)
        fig, ax = plt.subplots(figsize=(5.5, 4), dpi=160)
        ax.plot(recall, precision, label="PR")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{modality.upper()} PR Curve")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{modality}_pr.png")
        plt.close(fig)
    else:
        _save_placeholder_plot(
            fig_dir / f"{modality}_roc.png",
            f"{modality.upper()} ROC Curve",
            "ROC not available: only one class present in labels.",
        )
        _save_placeholder_plot(
            fig_dir / f"{modality}_pr.png",
            f"{modality.upper()} PR Curve",
            "PR not available: only one class present in labels.",
        )

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Tuple[float, float, float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    losses = []
    all_probs = []
    all_labels = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = forward_batch(model, x)
        loss = F.cross_entropy(logits, y)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.detach().cpu().numpy())
        losses.append(loss.item())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    metrics = evaluate_probs(labels, probs)
    return float(np.mean(losses)), metrics["acc"], metrics["auc"], metrics["ap"], metrics["s95"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResNet50 on scalograms")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_cfg(Path(args.config))
    index_dir = Path(cfg["index_dir"]).expanduser().resolve()
    modality = cfg["modality"]

    tiny_run = bool(cfg.get("tiny_run", False))
    max_train = int(cfg["max_train_samples"]) if tiny_run else None
    max_val = int(cfg["max_val_samples"]) if tiny_run else None
    max_test = int(cfg["max_test_samples"]) if tiny_run else None
    if tiny_run:
        print(
            f"[TINY RUN] Using tiny subset: max {max_train} train, {max_val} val, {max_test} test samples "
            "(same params & pipeline — for quick sanity check; run full data on GPU/Colab)."
        )

    train_loader = make_loader(
        cfg,
        index_dir / f"{modality}_train.csv",
        shuffle=True,
        is_train=True,
        max_samples=max_train,
    )
    val_loader = make_loader(
        cfg,
        index_dir / f"{modality}_val.csv",
        shuffle=False,
        is_train=False,
        max_samples=max_val,
    )
    test_loader = make_loader(
        cfg,
        index_dir / f"{modality}_test.csv",
        shuffle=False,
        is_train=False,
        max_samples=max_test,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = build_model().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    epochs = int(cfg["epochs"])
    patience = int(cfg.get("patience", 3))
    min_delta = float(cfg.get("min_delta", 0.001))
    scheduler_cfg = cfg.get("scheduler", {})
    scheduler = None
    if scheduler_cfg.get("name") == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(scheduler_cfg.get("mode", "max")),
            factor=float(scheduler_cfg.get("factor", 0.5)),
            patience=int(scheduler_cfg.get("patience", 2)),
            min_lr=float(scheduler_cfg.get("min_lr", 1e-6)),
        )
    best_val_auc = -1.0
    bad_epochs = 0
    history: list[dict] = []

    out_dir = Path(cfg["output_dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_run_metadata(out_dir, modality, cfg, device)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_auc, train_ap, train_s95 = run_epoch(
            model, train_loader, optimizer, device
        )
        val_loss, val_acc, val_auc, val_ap, val_s95 = run_epoch(model, val_loader, None, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_auc": train_auc,
                "train_ap": train_ap,
                "train_s95": train_s95,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_auc": val_auc,
                "val_ap": val_ap,
                "val_s95": val_s95,
            }
        )
        print(
            f"Epoch {epoch}/{epochs} "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} auc {train_auc:.4f} ap {train_ap:.4f} s95 {train_s95:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} auc {val_auc:.4f} ap {val_ap:.4f} s95 {val_s95:.4f}"
        )
        if val_auc > best_val_auc + min_delta:
            best_val_auc = val_auc
            bad_epochs = 0
            torch.save(model.state_dict(), out_dir / f"resnet50_{modality}_best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (no val AUC improvement).")
                break
        if scheduler is not None:
            scheduler.step(val_auc)

    test_loss, test_acc, test_auc, test_ap, test_s95 = run_epoch(model, test_loader, None, device)
    print(
        f"Test loss {test_loss:.4f} acc {test_acc:.4f} "
        f"auc {test_auc:.4f} ap {test_ap:.4f} s95 {test_s95:.4f}"
    )
    # Save core artifacts
    torch.save(model.state_dict(), out_dir / f"resnet50_{modality}.pt")
    metrics = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_auc": test_auc,
        "test_ap": test_ap,
        "test_sens_at_spec_0.95": test_s95,
    }
    with (out_dir / f"metrics_{modality}.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    # Save history and figures
    save_history(out_dir, modality, history)
    test_probs, test_labels = predict_probs(model, test_loader, device)
    np.savez(out_dir / f"test_{modality}_probs_labels.npz", probs=test_probs, labels=test_labels)
    save_figures(out_dir, modality, history, test_labels, test_probs)


if __name__ == "__main__":
    main()
