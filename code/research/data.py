"""Immutable input loading and leakage-aware experimental partitions."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    Y: np.ndarray
    ids: np.ndarray
    feature_names: list
    target_names: list
    structures: list
    metrics: list
    source_path: str
    source_sha256: str
    provenance: dict

    def subset(self, keep):
        return Dataset(self.name, self.X[keep], self.Y[keep], self.ids[keep],
                       self.feature_names, self.target_names, self.structures,
                       self.metrics, self.source_path, self.source_sha256, self.provenance)


def load_dataset(root, name, exclude_ids=(), cfd_variant="excel"):
    path = Path(root) / "data" / "raw" / (
        "user_confirmed_table_2026-11-09.json" if name == "real" else "cfd_legacy_snapshot.json")
    content = path.read_bytes()
    raw = json.loads(content.decode("utf-8-sig"))
    rows = np.asarray(raw["rows"], float)
    provenance = dict(raw.get("provenance", raw.get("source", {})))
    if name == "real":
        X, Y = rows[:, 1:5], rows[:, 5:]
        labels = [s["label"] for s in raw["structures"]]
        structures = [s for s in labels for _ in range(2)]
        metrics = [m for _ in labels for m in ("CD Loss", "AR")]
        names = [f"{s} {m}" for s, m in zip(structures, metrics)]
        feature_names = ["Coil", "Platen", "SF6", "Passivation pressure"]
    else:
        if cfd_variant not in {"excel", "legacy_code"}:
            raise ValueError(cfd_variant)
        if cfd_variant == "legacy_code":
            for difference in provenance["legacy_code_value_differences"]:
                index = difference["excel_row"] - 2
                if rows[index, 3] != difference["workbook_value"]:
                    raise ValueError("CFD discrepancy mapping no longer matches snapshot")
                rows[index, 3] = difference["code_value"]
        provenance["numerical_variant_used"] = cfd_variant
        provenance["snapshot_modified_on_disk"] = False
        X, Y = rows[:, 1:3], rows[:, 3:4]
        structures, metrics, names = ["CFD"], ["simulated response"], ["CFD response"]
        feature_names = ["Pressure", "Power"]
    if not np.isfinite(rows).all() or len(set(rows[:, 0])) != len(rows):
        raise ValueError("Input contains missing values or duplicate experiment IDs")
    result = Dataset(name, X, Y, rows[:, 0].astype(int), feature_names, names,
                     structures, metrics, str(path), hashlib.sha256(content).hexdigest(),
                     provenance)
    return result.subset(~np.isin(result.ids, list(exclude_ids)))


def source_groups(dataset):
    """2/13 share response-source evidence; never split across train/test.

    This is a conservative source block, not a claim that recipes are identical.
    """
    groups = dataset.ids.copy()
    if dataset.name == "real":
        groups[np.isin(dataset.ids, [2, 13])] = 2
    return groups


def make_split(dataset, seed, n_test, n_initial, budget):
    if n_test + n_initial + budget > len(dataset.ids):
        raise ValueError("Budget exceeds available candidate experiments")
    rng = np.random.default_rng(seed)
    groups = source_groups(dataset)
    # Rejection sampling preserves exactly requested test size while respecting
    # the predeclared 2/13 source block. All strategies reuse the same partition.
    for attempt in range(10000):
        order = rng.permutation(len(dataset.ids))
        test, remaining = order[:n_test], order[n_test:]
        if not set(groups[test]) & set(groups[remaining]):
            break
    else:
        raise RuntimeError("Could not construct requested source-blocked split")
    initial = np.sort(remaining[:n_initial])
    pool = np.sort(remaining[n_initial:])
    return {"test": np.sort(test), "initial": initial, "pool": pool,
            "rejection_attempts": attempt}


class LabelOracle:
    """Only reveal explicitly queried candidate experiment IDs."""
    def __init__(self, dataset, initial, pool, test):
        self._dataset = dataset
        self.revealed = set(map(int, initial))
        self.candidates = set(map(int, pool))
        self.test = set(map(int, test))
        if self.revealed & self.test or self.candidates & self.test:
            raise ValueError("Training/pool overlap with held-out test")

    def training(self):
        idx = np.array(sorted(self.revealed), dtype=int)
        return idx, self._dataset.X[idx], self._dataset.Y[idx]

    def reveal(self, index):
        index = int(index)
        if index not in self.candidates or index in self.test:
            raise ValueError("Only a remaining non-test candidate may be revealed")
        self.candidates.remove(index)
        self.revealed.add(index)
