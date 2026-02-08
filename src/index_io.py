from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import List

from .dataset import IndexItem


def load_index_csv(path: Path, max_samples: int | None = None, seed: int = 42) -> List[IndexItem]:
    items: List[IndexItem] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(IndexItem(Path(row["path"]), int(row["label"]), int(row["offset"])))
    if max_samples is None:
        return items
    # Stratified subsample so tiny run has both classes when possible
    by_label: dict[int, list[IndexItem]] = defaultdict(list)
    for it in items:
        by_label[it.label].append(it)
    n_per_class = max(1, max_samples // max(1, len(by_label)))
    out: List[IndexItem] = []
    for label in sorted(by_label.keys()):
        out.extend(by_label[label][:n_per_class])
    rng = random.Random(seed)
    rng.shuffle(out)
    return out[:max_samples]
