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
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

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


def make_loader(cfg: dict, index_csv: Path, shuffle: bool, max_samples: int | None = None) -> DataLoader:
    scales = np.array(cfg["scales"], dtype=np.float32)
    seed = int(cfg.get("seed", 42))
    index = load_index_csv(index_csv, max_samples=max_samples, seed=seed)
    dataset = ScalogramDataset(
        index=index,
        window_size=int(cfg["window_size"]),
        missing_value=float(cfg["missing_value"]),
        downsample_factor=int(cfg.get("downsample_factor", 1)),
        scales=scales,
        wavelet=str(cfg["wavelet"]),
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
    preds = (probs >= 0.5).astype(np.int64)
    acc = float(accuracy_score(labels, preds))
    auc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0
    ap = float(average_precision_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0
    s95 = sensitivity_at_specificity(labels, probs, target_spec=0.95) if len(np.unique(labels)) > 1 else 0.0
    return float(np.mean(losses)), acc, auc, ap, s95


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

    train_loader = make_loader(cfg, index_dir / f"{modality}_train.csv", shuffle=True, max_samples=max_train)
    val_loader = make_loader(cfg, index_dir / f"{modality}_val.csv", shuffle=False, max_samples=max_val)
    test_loader = make_loader(cfg, index_dir / f"{modality}_test.csv", shuffle=False, max_samples=max_test)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = build_model().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    epochs = int(cfg["epochs"])
    patience = int(cfg.get("patience", 3))
    min_delta = float(cfg.get("min_delta", 0.001))
    best_val_auc = -1.0
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_auc, train_ap, train_s95 = run_epoch(
            model, train_loader, optimizer, device
        )
        val_loss, val_acc, val_auc, val_ap, val_s95 = run_epoch(model, val_loader, None, device)
        print(
            f"Epoch {epoch}/{epochs} "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} auc {train_auc:.4f} ap {train_ap:.4f} s95 {train_s95:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} auc {val_auc:.4f} ap {val_ap:.4f} s95 {val_s95:.4f}"
        )
        if val_auc > best_val_auc + min_delta:
            best_val_auc = val_auc
            bad_epochs = 0
            out_dir = Path(cfg["output_dir"]).expanduser().resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_dir / f"resnet50_{modality}_best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (no val AUC improvement).")
                break

    test_loss, test_acc, test_auc, test_ap, test_s95 = run_epoch(model, test_loader, None, device)
    print(
        f"Test loss {test_loss:.4f} acc {test_acc:.4f} "
        f"auc {test_auc:.4f} ap {test_ap:.4f} s95 {test_s95:.4f}"
    )

    out_dir = Path(cfg["output_dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
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


if __name__ == "__main__":
    main()
