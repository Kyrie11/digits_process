from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from .utils import ensure_dir, save_json

FEATURE_KEYS = ["digits_vec", "X", "images", "data", "fea", "features", "digitData", "digits", "imgs"]
LABEL_KEYS = ["digits_labels", "y", "Y", "labels", "label", "gnd", "target"]
TRAIN_KEYS = ["trainset", "train_set", "trainIdx", "train_idx", "train_index"]
TEST_KEYS = ["testset", "test_set", "testIdx", "test_idx", "test_index"]


@dataclass
class DatasetBundle:
    X: np.ndarray
    y: np.ndarray
    train_splits: list[np.ndarray]
    test_splits: list[np.ndarray]
    class_names: list[str]
    original_labels: np.ndarray
    feature_key: str
    label_key: str
    train_key: str
    test_key: str

    @property
    def num_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def num_trials(self) -> int:
        return len(self.train_splits)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def _pick_key(payload: dict[str, Any], candidates: list[str], field_name: str) -> str:
    for key in candidates:
        if key in payload:
            return key
    available = [k for k in payload.keys() if not k.startswith("__")]
    raise KeyError(f"Could not find {field_name}. Tried {candidates}. Available keys: {available}")


def _coerce_labels(y_raw: Any) -> tuple[np.ndarray, np.ndarray, list[str]]:
    y = np.asarray(y_raw).squeeze()
    if y.ndim != 1:
        y = y.reshape(-1)

    if np.issubdtype(y.dtype, np.floating):
        y = np.round(y)
    original = y.astype(np.int64)

    uniq = np.unique(original)
    label_to_idx = {int(label): idx for idx, label in enumerate(uniq.tolist())}
    y_encoded = np.array([label_to_idx[int(v)] for v in original], dtype=np.int64)
    class_names = [str(int(v)) for v in uniq.tolist()]
    return y_encoded, original, class_names


def _coerce_features(X_raw: Any, n_samples_hint: int | None = None) -> np.ndarray:
    X = np.asarray(X_raw)
    if X.ndim != 2:
        raise ValueError(f"Feature matrix must be 2D, got shape={X.shape}")

    if n_samples_hint is not None:
        if X.shape[0] == n_samples_hint:
            pass
        elif X.shape[1] == n_samples_hint:
            X = X.T
        else:
            raise ValueError(
                f"Feature matrix shape {X.shape} is inconsistent with label length {n_samples_hint}."
            )
    else:
        if X.shape[0] < X.shape[1]:
            X = X.T

    X = X.astype(np.float32)
    return X


def _normalize_to_unit_range(X: np.ndarray) -> np.ndarray:
    x_min = float(X.min())
    x_max = float(X.max())
    if x_min >= 0.0 and x_max <= 1.0 + 1e-6:
        return np.clip(X, 0.0, 1.0)
    if x_min >= 0.0 and x_max <= 255.0 + 1e-6:
        return np.clip(X / 255.0, 0.0, 1.0)
    denom = max(x_max - x_min, 1e-12)
    return np.clip((X - x_min) / denom, 0.0, 1.0)


def _normalize_indices(values: np.ndarray, n_samples: int) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    if np.issubdtype(values.dtype, np.floating):
        values = values[np.isfinite(values)]
        values = np.round(values)
    idx = values.astype(np.int64)
    if idx.size == 0:
        return idx
    if idx.min() >= 1 and idx.max() <= n_samples:
        idx = idx - 1  # MATLAB -> Python
    if idx.min() < 0 or idx.max() >= n_samples:
        raise ValueError(f"Split indices out of range: min={idx.min()}, max={idx.max()}, n={n_samples}")
    return idx


def _parse_split_matrix(split_raw: Any, n_samples: int) -> list[np.ndarray]:
    arr = np.asarray(split_raw)

    if arr.dtype == object:
        trials = []
        for item in arr.reshape(-1):
            trials.append(_normalize_indices(np.asarray(item).reshape(-1), n_samples))
        return trials

    arr = np.squeeze(arr)
    if arr.ndim == 1:
        return [_normalize_indices(arr, n_samples)]
    if arr.ndim != 2:
        raise ValueError(f"Split array must be 1D or 2D, got shape={arr.shape}")

    if arr.shape[0] <= 10 and arr.shape[0] <= arr.shape[1]:
        return [_normalize_indices(arr[i, :], n_samples) for i in range(arr.shape[0])]
    if arr.shape[1] <= 10 and arr.shape[1] < arr.shape[0]:
        return [_normalize_indices(arr[:, j], n_samples) for j in range(arr.shape[1])]

    if arr.shape[0] <= arr.shape[1]:
        return [_normalize_indices(arr[i, :], n_samples) for i in range(arr.shape[0])]
    return [_normalize_indices(arr[:, j], n_samples) for j in range(arr.shape[1])]


def load_digits_mat(mat_path: str | Path, auto_normalize_01: bool = True) -> DatasetBundle:
    mat_path = Path(mat_path)
    payload = loadmat(mat_path)

    label_key = _pick_key(payload, LABEL_KEYS, "labels")
    y_encoded, original_labels, class_names = _coerce_labels(payload[label_key])
    feature_key = _pick_key(payload, FEATURE_KEYS, "features")
    X = _coerce_features(payload[feature_key], n_samples_hint=len(y_encoded))

    if auto_normalize_01:
        X = _normalize_to_unit_range(X)

    train_key = _pick_key(payload, TRAIN_KEYS, "train splits")
    test_key = _pick_key(payload, TEST_KEYS, "test splits")
    train_splits = _parse_split_matrix(payload[train_key], X.shape[0])
    test_splits = _parse_split_matrix(payload[test_key], X.shape[0])

    if len(train_splits) != len(test_splits):
        raise ValueError(f"Number of train/test trials mismatch: {len(train_splits)} vs {len(test_splits)}")

    return DatasetBundle(
        X=X,
        y=y_encoded,
        train_splits=train_splits,
        test_splits=test_splits,
        class_names=class_names,
        original_labels=original_labels,
        feature_key=feature_key,
        label_key=label_key,
        train_key=train_key,
        test_key=test_key,
    )


def get_trial_data(bundle: DatasetBundle, trial_idx: int):
    train_idx = bundle.train_splits[trial_idx]
    test_idx = bundle.test_splits[trial_idx]
    X_train = bundle.X[train_idx]
    y_train = bundle.y[train_idx]
    X_test = bundle.X[test_idx]
    y_test = bundle.y[test_idx]
    return X_train, y_train, X_test, y_test, train_idx, test_idx


def dataset_info(bundle: DatasetBundle, mat_path: str | Path) -> dict[str, Any]:
    return {
        "mat_path": str(mat_path),
        "num_samples": bundle.num_samples,
        "num_features": bundle.num_features,
        "num_classes": bundle.num_classes,
        "class_names": bundle.class_names,
        "num_trials": bundle.num_trials,
        "feature_key": bundle.feature_key,
        "label_key": bundle.label_key,
        "train_key": bundle.train_key,
        "test_key": bundle.test_key,
        "train_sizes": [int(len(v)) for v in bundle.train_splits],
        "test_sizes": [int(len(v)) for v in bundle.test_splits],
        "pixel_min": float(bundle.X.min()),
        "pixel_max": float(bundle.X.max()),
        "pixel_mean": float(bundle.X.mean()),
        "pixel_std": float(bundle.X.std()),
    }


def save_dataset_inspection(bundle: DatasetBundle, mat_path: str | Path, output_dir: str | Path, max_images: int = 25) -> None:
    output_dir = ensure_dir(output_dir)
    info = dataset_info(bundle, mat_path)
    save_json(info, output_dir / "dataset_info.json")

    unique, counts = np.unique(bundle.y, return_counts=True)
    labels = [bundle.class_names[int(i)] for i in unique.tolist()]

    plt.figure(figsize=(8, 4))
    plt.bar(labels, counts)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Label distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "label_distribution.png", dpi=200)
    plt.close()

    n = min(max_images, bundle.num_samples)
    side = int(np.sqrt(bundle.num_features))
    cols = 5
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 2.2, rows * 2.2))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(bundle.X[i].reshape(side, side), cmap="gray")
        plt.title(f"y={bundle.class_names[bundle.y[i]]}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "dataset_preview.png", dpi=200)
    plt.close()
