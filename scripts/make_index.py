from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_io import SignalFile, iter_segments, list_signal_files, load_mat_signal


@dataclass(frozen=True)
class FileSplit:
    train: List[SignalFile]
    val: List[SignalFile]
    test: List[SignalFile]


def split_files(
    files: List[SignalFile],
    test_size: float,
    val_size: float,
    seed: int,
) -> FileSplit:
    labels = np.array([f.label for f in files])
    train_files, test_files = train_test_split(
        files, test_size=test_size, random_state=seed, stratify=labels
    )
    train_labels = np.array([f.label for f in train_files])
    val_ratio = val_size / (1.0 - test_size)
    train_files, val_files = train_test_split(
        train_files, test_size=val_ratio, random_state=seed, stratify=train_labels
    )
    return FileSplit(train=train_files, val=val_files, test=test_files)


def subject_id_from_path(path: Path) -> str:
    # Filenames like s01840-3454-10-24-18-46m.mat -> subject id s01840
    return path.stem.split("-")[0]


def split_by_subject(
    files: List[SignalFile],
    test_size: float,
    val_size: float,
    seed: int,
) -> FileSplit:
    subjects: Dict[str, List[SignalFile]] = {}
    subject_labels: Dict[str, int] = {}
    for sf in files:
        sid = subject_id_from_path(sf.path)
        subjects.setdefault(sid, []).append(sf)
        subject_labels.setdefault(sid, sf.label)

    subj_ids = list(subjects.keys())
    subj_labels = np.array([subject_labels[sid] for sid in subj_ids])

    train_ids, test_ids = train_test_split(
        subj_ids, test_size=test_size, random_state=seed, stratify=subj_labels
    )
    train_labels = np.array([subject_labels[sid] for sid in train_ids])
    val_ratio = val_size / (1.0 - test_size)
    train_ids, val_ids = train_test_split(
        train_ids, test_size=val_ratio, random_state=seed, stratify=train_labels
    )

    def expand(ids: List[str]) -> List[SignalFile]:
        out: List[SignalFile] = []
        for sid in ids:
            out.extend(subjects[sid])
        return out

    return FileSplit(train=expand(train_ids), val=expand(val_ids), test=expand(test_ids))


def build_rows(
    files: List[SignalFile],
    window_size: int,
    stride: int,
    missing_value: float,
    max_segments_per_file: int,
) -> List[Tuple[str, int, int]]:
    rows: List[Tuple[str, int, int]] = []
    for sf in files:
        signal = load_mat_signal(sf.path)
        starts = [start for start, _ in iter_segments(signal, window_size, stride, missing_value)]
        if max_segments_per_file > 0 and len(starts) > max_segments_per_file:
            idx = np.linspace(0, len(starts) - 1, max_segments_per_file).astype(int)
            starts = [starts[i] for i in idx]
        for start in starts:
            rows.append((str(sf.path), sf.label, start))
    return rows


def write_csv(path: Path, rows: List[Tuple[str, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "offset"])
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build segment index CSVs")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config (e.g., configs/abp.yaml)",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["data_root"]).expanduser().resolve()
    modality = cfg["modality"]
    window_size = int(cfg["window_size"])
    stride = int(cfg["stride"])
    missing_value = float(cfg["missing_value"])
    max_segments_per_file = int(cfg.get("max_segments_per_file", 0))
    test_size = float(cfg["test_size"])
    val_size = float(cfg["val_size"])
    seed = int(cfg["seed"])
    out_dir = Path(cfg["index_dir"]).expanduser().resolve()

    files = list_signal_files(root, modality)
    split = split_by_subject(files, test_size, val_size, seed)

    write_csv(
        out_dir / f"{modality}_train.csv",
        build_rows(split.train, window_size, stride, missing_value, max_segments_per_file),
    )
    write_csv(
        out_dir / f"{modality}_val.csv",
        build_rows(split.val, window_size, stride, missing_value, max_segments_per_file),
    )
    write_csv(
        out_dir / f"{modality}_test.csv",
        build_rows(split.test, window_size, stride, missing_value, max_segments_per_file),
    )

    print(
        f"Saved index CSVs to {out_dir} for {modality} "
        f"(train {len(split.train)} files, val {len(split.val)} files, test {len(split.test)} files)"
    )


if __name__ == "__main__":
    main()
