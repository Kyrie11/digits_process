from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.model_selection import ParameterGrid, StratifiedKFold

from .metrics import accuracy_score
from .models import create_model
from .preprocess import build_preprocessor


def resolve_cv_folds(y: np.ndarray, requested_folds: int) -> int:
    _, counts = np.unique(y, return_counts=True)
    max_allowed = int(counts.min())
    folds = min(int(requested_folds), max_allowed)
    if folds < 2:
        raise ValueError(
            f"Not enough samples per class for cross-validation. Requested={requested_folds}, min_count={max_allowed}"
        )
    return folds


def cross_validate_configuration(
    X_train: np.ndarray,
    y_train: np.ndarray,
    preprocess_method: str,
    pca_dim: int | None,
    model_name: str,
    model_cfg: dict[str, Any],
    cv_folds: int,
    device: str,
    seed: int,
    num_classes: int,
):
    grid = model_cfg.get("grid", {})
    param_list = list(ParameterGrid(grid)) if grid else [{}]
    folds = resolve_cv_folds(y_train, cv_folds)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    split_indices = list(splitter.split(X_train, y_train))

    cached_folds = []
    for fold_id, (tr_idx, val_idx) in enumerate(split_indices):
        X_tr = torch.as_tensor(X_train[tr_idx], dtype=torch.float32)
        X_val = torch.as_tensor(X_train[val_idx], dtype=torch.float32)
        y_tr = y_train[tr_idx]
        y_val = y_train[val_idx]

        preprocessor = build_preprocessor(preprocess_method, pca_dim)
        preprocessor.fit(X_tr)
        X_tr_t = preprocessor.transform(X_tr).cpu().numpy()
        X_val_t = preprocessor.transform(X_val).cpu().numpy()
        cached_folds.append((fold_id, X_tr_t, y_tr, X_val_t, y_val))

    all_records = []
    best_record = None

    for param_idx, params in enumerate(param_list):
        fold_scores = []
        for fold_id, X_tr_t, y_tr, X_val_t, y_val in cached_folds:
            model = create_model(
                model_name=model_name,
                input_dim=X_tr_t.shape[1],
                num_classes=num_classes,
                hyperparams=params,
                train_cfg=model_cfg.get("train", {}),
                device=device,
                seed=seed + param_idx * 100 + fold_id,
            )
            model.fit(X_tr_t, y_tr)
            pred = model.predict(X_val_t)
            fold_scores.append(accuracy_score(y_val, pred))

        record = {
            "params": params,
            "fold_scores": [float(v) for v in fold_scores],
            "mean_cv_accuracy": float(np.mean(fold_scores)),
            "std_cv_accuracy": float(np.std(fold_scores)),
            "cv_folds_used": folds,
        }
        all_records.append(record)
        if best_record is None or record["mean_cv_accuracy"] > best_record["mean_cv_accuracy"] or (
            np.isclose(record["mean_cv_accuracy"], best_record["mean_cv_accuracy"]) and
            record["std_cv_accuracy"] < best_record["std_cv_accuracy"]
        ):
            best_record = record

    assert best_record is not None
    return best_record, all_records
