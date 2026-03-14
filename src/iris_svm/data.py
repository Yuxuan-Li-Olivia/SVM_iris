from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

FEATURE_COLS_DEFAULT = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]
TARGET_COL_DEFAULT = "species"


@dataclass(frozen=True)
class Dataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    class_names: List[str]


def _load_with_pandas(path: Path, feature_cols: List[str], target_col: str) -> Dataset:
    import pandas as pd  # type: ignore

    df = pd.read_csv(path)
    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. Found: {list(df.columns)}")

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].astype(str).to_numpy()
    class_names = sorted(np.unique(y).tolist())
    return Dataset(X=X, y=y, feature_names=feature_cols, class_names=class_names)


def _load_with_csv_module(path: Path, feature_cols: List[str], target_col: str) -> Dataset:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")

        missing = [c for c in feature_cols + [target_col] if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}. Found: {reader.fieldnames}")

        X_rows: List[List[float]] = []
        y_rows: List[str] = []
        for row in reader:
            X_rows.append([float(row[c]) for c in feature_cols])
            y_rows.append(str(row[target_col]))

    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows, dtype=str)
    class_names = sorted(np.unique(y).tolist())
    return Dataset(X=X, y=y, feature_names=feature_cols, class_names=class_names)


def load_dataset(path: Path, feature_cols: List[str] | None = None, target_col: str = TARGET_COL_DEFAULT) -> Dataset:
    """Load iris dataset.

    Uses pandas if available, otherwise falls back to Python's csv module.
    """
    feature_cols = feature_cols or FEATURE_COLS_DEFAULT

    try:
        return _load_with_pandas(path, feature_cols, target_col)
    except ModuleNotFoundError:
        return _load_with_csv_module(path, feature_cols, target_col)
