from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_io import clean_segment, load_mat_signal
from src.index_io import load_index_csv


def sensitivity_at_specificity(labels: np.ndarray, probs: np.ndarray, target_spec: float) -> float:
    fpr, tpr, _ = roc_curve(labels, probs)
    spec = 1.0 - fpr
    mask = spec >= target_spec
    if not np.any(mask):
        return 0.0
    return float(np.max(tpr[mask]))


def basic_features(x: np.ndarray) -> np.ndarray:
    # Basic time-domain statistics + simple spectral summaries
    mean = np.mean(x)
    std = np.std(x)
    median = np.median(x)
    min_v = np.min(x)
    max_v = np.max(x)
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    rms = np.sqrt(np.mean(x * x))
    zcr = np.mean(x[1:] * x[:-1] < 0)

    fft = np.fft.rfft(x)
    power = np.abs(fft) ** 2
    total = np.sum(power) + 1e-8
    bins = power.shape[0]
    third = bins // 3
    e1 = np.sum(power[:third]) / total
    e2 = np.sum(power[third : 2 * third]) / total
    e3 = np.sum(power[2 * third :]) / total
    centroid = np.sum(np.arange(bins) * power) / total

    return np.array(
        [mean, std, median, min_v, max_v, iqr, rms, zcr, e1, e2, e3, centroid],
        dtype=np.float32,
    )


def build_xy(
    index_csv: Path,
    window_size: int,
    missing_value: float,
    max_samples: int | None = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    items = load_index_csv(index_csv, max_samples=max_samples, seed=seed)
    x_list = []
    y_list = []
    for item in items:
        signal = load_mat_signal(item.path)
        segment = signal[item.offset : item.offset + window_size]
        segment = clean_segment(segment, missing_value)
        x_list.append(basic_features(segment))
        y_list.append(item.label)
    return np.stack(x_list), np.array(y_list, dtype=np.int64)


def evaluate(labels: np.ndarray, probs: np.ndarray) -> dict:
    preds = (probs >= 0.5).astype(np.int64)
    acc = float(accuracy_score(labels, preds))
    auc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0
    ap = float(average_precision_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0
    s95 = sensitivity_at_specificity(labels, probs, 0.95) if len(np.unique(labels)) > 1 else 0.0
    return {"acc": acc, "auc": auc, "ap": ap, "s95": s95}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run classical feature baselines")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    index_dir = Path(cfg["index_dir"]).expanduser().resolve()
    modality = cfg["modality"]
    window_size = int(cfg["window_size"])
    missing_value = float(cfg["missing_value"])

    tiny_run = bool(cfg.get("tiny_run", False))
    max_train = int(cfg["max_train_samples"]) if tiny_run else None
    max_val = int(cfg["max_val_samples"]) if tiny_run else None
    max_test = int(cfg["max_test_samples"]) if tiny_run else None
    if tiny_run:
        print(
            f"[TINY RUN] Using tiny subset: max {max_train} train, {max_val} val, {max_test} test "
            "(same params — quick sanity check)."
        )

    seed = int(cfg.get("seed", 42))
    x_train, y_train = build_xy(
        index_dir / f"{modality}_train.csv", window_size, missing_value, max_samples=max_train, seed=seed
    )
    x_val, y_val = build_xy(
        index_dir / f"{modality}_val.csv", window_size, missing_value, max_samples=max_val, seed=seed
    )
    x_test, y_test = build_xy(
        index_dir / f"{modality}_test.csv", window_size, missing_value, max_samples=max_test, seed=seed
    )

    results = {}

    lr = LogisticRegression(max_iter=200, n_jobs=1)
    lr.fit(x_train, y_train)
    lr_probs = lr.predict_proba(x_test)[:, 1]
    results["logreg"] = evaluate(y_test, lr_probs)

    rf = RandomForestClassifier(
        n_estimators=200, random_state=cfg["seed"], n_jobs=1, max_depth=None
    )
    rf.fit(x_train, y_train)
    rf_probs = rf.predict_proba(x_test)[:, 1]
    results["random_forest"] = evaluate(y_test, rf_probs)

    out_dir = Path(cfg["output_dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"baseline_{modality}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved baseline results to {out_path}")


if __name__ == "__main__":
    main()
