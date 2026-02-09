from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml


def subject_id_from_path(path: str) -> str:
    # Filenames like s01840-3454-10-24-18-46m.mat -> subject id s01840
    return Path(path).stem.split("-")[0]


def read_index(path: Path) -> List[dict]:
    rows = []
    with path.open("r") as f:
        header = f.readline()
        for line in f:
            p, label, offset = line.strip().split(",")
            rows.append({"path": p, "label": int(label), "offset": int(offset)})
    return rows


def summarize_index(rows: List[dict]) -> Dict[str, int]:
    labels = [r["label"] for r in rows]
    subjects = {subject_id_from_path(r["path"]) for r in rows}
    return {
        "segments": len(rows),
        "cad_segments": int(sum(labels)),
        "noncad_segments": int(len(labels) - sum(labels)),
        "subjects": len(subjects),
    }


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_history(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def best_epoch_by_val_auc(history: List[dict]) -> dict | None:
    if not history:
        return None
    return max(history, key=lambda h: h.get("val_auc", -1.0))


def modality_section(cfg: dict) -> str:
    modality = cfg["modality"]
    index_dir = Path(cfg["index_dir"]).expanduser().resolve()
    output_dir = Path(cfg["output_dir"]).expanduser().resolve()

    train_rows = read_index(index_dir / f"{modality}_train.csv")
    val_rows = read_index(index_dir / f"{modality}_val.csv")
    test_rows = read_index(index_dir / f"{modality}_test.csv")

    train_stats = summarize_index(train_rows)
    val_stats = summarize_index(val_rows)
    test_stats = summarize_index(test_rows)

    metrics = load_json(output_dir / f"metrics_{modality}.json")
    baseline = load_json(output_dir / f"baseline_{modality}.json")
    history = load_history(output_dir / f"history_{modality}.json")
    run_meta = load_json(output_dir / f"run_{modality}.json")
    best = best_epoch_by_val_auc(history)

    figures = [
        output_dir / "figures" / f"{modality}_loss.png",
        output_dir / "figures" / f"{modality}_auc.png",
        output_dir / "figures" / f"{modality}_acc.png",
        output_dir / "figures" / f"{modality}_ap.png",
        output_dir / "figures" / f"{modality}_roc.png",
        output_dir / "figures" / f"{modality}_pr.png",
        output_dir / "figures" / f"{modality}_cm.png",
    ]

    fig_list = "\n".join([f"- `{p}`" for p in figures if p.exists()])
    fig_list = fig_list or "- (figures not found; run training first)"

    device = run_meta.get("device") if run_meta else "unknown"

    lines = [
        f"## {modality.upper()}",
        "",
        "### Dataset statistics",
        f"- Train segments: {train_stats['segments']} (CAD {train_stats['cad_segments']}, NONCAD {train_stats['noncad_segments']}), subjects: {train_stats['subjects']}",
        f"- Val segments: {val_stats['segments']} (CAD {val_stats['cad_segments']}, NONCAD {val_stats['noncad_segments']}), subjects: {val_stats['subjects']}",
        f"- Test segments: {test_stats['segments']} (CAD {test_stats['cad_segments']}, NONCAD {test_stats['noncad_segments']}), subjects: {test_stats['subjects']}",
        "",
        "### Methodology",
        f"- Window size: {cfg['window_size']} samples",
        f"- Stride: {cfg['stride']} samples",
        f"- Downsample factor: {cfg.get('downsample_factor', 1)}",
        f"- Missing value: {cfg['missing_value']} (filled with segment median)",
        f"- CWT wavelet: {cfg['wavelet']}",
        f"- CWT scales: {len(cfg['scales'])} (min {min(cfg['scales'])}, max {max(cfg['scales'])})",
        f"- Scalogram normalization: per-sample min-max to [0,1], stacked to 3 channels, resized to 224×224; model input normalization={cfg.get('normalize', 'none')}",
        f"- Model: ResNet50 (ImageNet pretrained), 2-class head",
        f"- Augmentation: {cfg.get('augment', False)} (time shift max={cfg.get('time_shift_max', 0)}, noise_std={cfg.get('noise_std', 0.0)})",
        f"- Batch size: {cfg['batch_size']}",
        f"- Optimizer: Adam, lr={cfg['lr']}",
        f"- Epochs: {cfg['epochs']} (early stopping: patience={cfg.get('patience', 3)}, min_delta={cfg.get('min_delta', 0.001)})",
        f"- Device: {device}",
        "",
        "### Results",
    ]

    if metrics:
        lines += [
            f"- Test loss: {metrics['test_loss']:.4f}",
            f"- Test acc: {metrics['test_acc']:.4f}",
            f"- Test AUROC: {metrics['test_auc']:.4f}",
            f"- Test AUPRC: {metrics['test_ap']:.4f}",
            f"- Test sensitivity @ 0.95 specificity: {metrics['test_sens_at_spec_0.95']:.4f}",
        ]
    else:
        lines.append("- Test metrics not found.")

    if best:
        lines.append(
            f"- Best val AUROC: {best.get('val_auc', 0.0):.4f} at epoch {best.get('epoch', '-')}"
        )

    if baseline:
        lines += [
            "",
            "### Baselines",
            f"- Logistic Regression: acc {baseline['logreg']['acc']:.4f}, AUC {baseline['logreg']['auc']:.4f}, AP {baseline['logreg']['ap']:.4f}",
            f"- Random Forest: acc {baseline['random_forest']['acc']:.4f}, AUC {baseline['random_forest']['auc']:.4f}, AP {baseline['random_forest']['ap']:.4f}",
        ]

    lines += [
        "",
        "### Figures",
        fig_list,
        "",
    ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate methods/results report from outputs")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["configs/abp.yaml", "configs/ppg.yaml"],
        help="Config files to include",
    )
    parser.add_argument("--out", type=str, default="docs/REPORT.md")
    args = parser.parse_args()

    sections = []
    for cfg_path in args.configs:
        cfg = yaml.safe_load(Path(cfg_path).read_text())
        sections.append(modality_section(cfg))

    report = "\n".join(
        [
            "# W2CAD Methods & Results Report",
            "",
            "This report is auto-generated from configs and output artifacts.",
            "",
            *sections,
        ]
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"Wrote report to {out_path}")


if __name__ == "__main__":
    main()
