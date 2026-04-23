from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .crossval import cross_validate_configuration
from .data import get_trial_data, load_digits_mat, save_dataset_inspection
from .metrics import accuracy_score
from .models import create_model
from .preprocess import add_gaussian_noise, build_preprocessor
from .utils import ensure_dir, get_device, resolve_output_dir, save_json, save_yaml, set_seed, slugify


def inspect_command(cfg: dict[str, Any]) -> None:
    set_seed(int(cfg["experiment"].get("seed", 42)))
    output_dir = resolve_output_dir(cfg)
    inspection_dir = ensure_dir(output_dir / "inspection")
    mat_path = cfg["data"]["mat_path"]
    bundle = load_digits_mat(mat_path, auto_normalize_01=bool(cfg["data"].get("auto_normalize_01", True)))
    save_dataset_inspection(bundle, mat_path, inspection_dir)
    print(f"[inspect] Saved inspection assets to: {inspection_dir}")
    print(f"[inspect] num_samples={bundle.num_samples}, num_features={bundle.num_features}, num_trials={bundle.num_trials}")


def _save_prediction_artifact(
    pred_dir: Path,
    filename: str,
    y_true: np.ndarray,
    y_pred_clean: np.ndarray,
    noisy_preds: dict[float, np.ndarray],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
):
    path = pred_dir / filename
    noise_keys = sorted(float(k) for k in noisy_preds.keys())
    noise_levels = np.array(noise_keys, dtype=np.float32)
    noisy_matrix = np.stack([noisy_preds[k] for k in noise_keys], axis=0) if noisy_preds else np.empty((0, len(y_true)), dtype=np.int64)

    np.savez_compressed(
        path,
        y_true=y_true,
        y_pred_clean=y_pred_clean,
        noise_levels=noise_levels,
        noisy_preds=noisy_matrix,
        train_indices=train_indices,
        test_indices=test_indices,
    )
    return path


def run_command(cfg: dict[str, Any]) -> None:
    seed = int(cfg["experiment"].get("seed", 42))
    set_seed(seed)
    device = get_device(cfg["experiment"].get("device", "auto"))
    output_dir = resolve_output_dir(cfg)
    pred_dir = ensure_dir(output_dir / "prediction_artifacts")
    ensure_dir(output_dir / "tables")
    ensure_dir(output_dir / "figures")
    save_yaml(cfg, output_dir / "config_resolved.yaml")

    mat_path = cfg["data"]["mat_path"]
    bundle = load_digits_mat(mat_path, auto_normalize_01=bool(cfg["data"].get("auto_normalize_01", True)))

    preprocessing_cfg = cfg["preprocessing"]
    methods = preprocessing_cfg["methods"]
    pca_dims = preprocessing_cfg.get("pca_dimensions", [])
    noise_levels = [float(v) for v in cfg["evaluation"].get("noise_levels", [0.0])]
    model_space = cfg["models"]
    cv_folds = int(cfg["experiment"].get("cv_folds", 5))

    metrics_rows = []
    cv_rows = []

    trial_indices = cfg.get("experiment", {}).get("trial_indices")
    if trial_indices is None:
        trial_indices = list(range(bundle.num_trials))

    for trial_idx in trial_indices:
        trial_idx = int(trial_idx)
        X_train_raw, y_train, X_test_raw, y_test, train_idx, test_idx = get_trial_data(bundle, trial_idx)
        for preprocess_method in methods:
            dims = pca_dims if "pca" in preprocess_method else [None]
            for pca_dim in dims:
                for model_name, model_cfg in model_space.items():
                    print(
                        f"[run] trial={trial_idx} preprocess={preprocess_method} pca_dim={pca_dim} model={model_name}",
                        flush=True,
                    )
                    best_cv, all_cv_records = cross_validate_configuration(
                        X_train=X_train_raw,
                        y_train=y_train,
                        preprocess_method=preprocess_method,
                        pca_dim=pca_dim,
                        model_name=model_name,
                        model_cfg=model_cfg,
                        cv_folds=cv_folds,
                        device=device,
                        seed=seed + trial_idx,
                        num_classes=bundle.num_classes,
                    )

                    for cv_record in all_cv_records:
                        cv_rows.append(
                            {
                                "trial": trial_idx,
                                "preprocess": preprocess_method,
                                "pca_dim": pca_dim,
                                "model": model_name,
                                "params_json": json.dumps(cv_record["params"], ensure_ascii=False, sort_keys=True),
                                "mean_cv_accuracy": cv_record["mean_cv_accuracy"],
                                "std_cv_accuracy": cv_record["std_cv_accuracy"],
                                "cv_folds_used": cv_record["cv_folds_used"],
                            }
                        )

                    preprocessor = build_preprocessor(preprocess_method, pca_dim)
                    X_train_tensor = torch.as_tensor(X_train_raw, dtype=torch.float32)
                    X_test_clean_tensor = torch.as_tensor(X_test_raw, dtype=torch.float32)
                    preprocessor.fit(X_train_tensor)
                    X_train = preprocessor.transform(X_train_tensor).cpu().numpy()
                    X_test_clean = preprocessor.transform(X_test_clean_tensor).cpu().numpy()

                    model = create_model(
                        model_name=model_name,
                        input_dim=X_train.shape[1],
                        num_classes=bundle.num_classes,
                        hyperparams=best_cv["params"],
                        train_cfg=model_cfg.get("train", {}),
                        device=device,
                        seed=seed + trial_idx,
                    )
                    model.fit(X_train, y_train)
                    clean_pred = model.predict(X_test_clean)
                    clean_acc = accuracy_score(y_test, clean_pred)

                    noisy_preds: dict[float, np.ndarray] = {}
                    artifact_name = slugify(f"trial{trial_idx}", preprocess_method, f"pca{pca_dim}", model_name) + ".npz"
                    prediction_path = _save_prediction_artifact(
                        pred_dir=pred_dir,
                        filename=artifact_name,
                        y_true=y_test,
                        y_pred_clean=clean_pred,
                        noisy_preds=noisy_preds,
                        train_indices=train_idx,
                        test_indices=test_idx,
                    )

                    metrics_rows.append(
                        {
                            "trial": trial_idx,
                            "preprocess": preprocess_method,
                            "pca_dim": pca_dim,
                            "model": model_name,
                            "noise_sigma": 0.0,
                            "test_accuracy": clean_acc,
                            "cv_mean_accuracy": best_cv["mean_cv_accuracy"],
                            "cv_std_accuracy": best_cv["std_cv_accuracy"],
                            "best_params_json": json.dumps(best_cv["params"], ensure_ascii=False, sort_keys=True),
                            "feature_dim_after_transform": int(X_train.shape[1]),
                            "prediction_file": str(prediction_path.relative_to(output_dir)),
                        }
                    )

                    for sigma in noise_levels:
                        if float(sigma) == 0.0:
                            noisy_preds[0.0] = clean_pred
                            continue
                        noisy_test_raw = add_gaussian_noise(
                            X_test_raw,
                            sigma=float(sigma),
                            seed=seed + trial_idx * 1000 + int(100 * sigma),
                            clip_min=0.0,
                            clip_max=1.0,
                        )
                        noisy_test_transformed = preprocessor.transform(
                            torch.as_tensor(noisy_test_raw, dtype=torch.float32)
                        ).cpu().numpy()
                        noisy_pred = model.predict(noisy_test_transformed)
                        noisy_preds[float(sigma)] = noisy_pred
                        metrics_rows.append(
                            {
                                "trial": trial_idx,
                                "preprocess": preprocess_method,
                                "pca_dim": pca_dim,
                                "model": model_name,
                                "noise_sigma": float(sigma),
                                "test_accuracy": accuracy_score(y_test, noisy_pred),
                                "cv_mean_accuracy": best_cv["mean_cv_accuracy"],
                                "cv_std_accuracy": best_cv["std_cv_accuracy"],
                                "best_params_json": json.dumps(best_cv["params"], ensure_ascii=False, sort_keys=True),
                                "feature_dim_after_transform": int(X_train.shape[1]),
                                "prediction_file": str(prediction_path.relative_to(output_dir)),
                            }
                        )

                    if noisy_preds:
                        prediction_path.unlink(missing_ok=True)
                        prediction_path = _save_prediction_artifact(
                            pred_dir=pred_dir,
                            filename=artifact_name,
                            y_true=y_test,
                            y_pred_clean=clean_pred,
                            noisy_preds=noisy_preds,
                            train_indices=train_idx,
                            test_indices=test_idx,
                        )
                        for row in metrics_rows[-(1 + max(0, len(noise_levels) - 1)):]:
                            row["prediction_file"] = str(prediction_path.relative_to(output_dir))

    metrics_df = pd.DataFrame(metrics_rows)
    cv_df = pd.DataFrame(cv_rows)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    cv_df.to_csv(output_dir / "cv_results.csv", index=False)
    save_json({"device": device, "num_trials": bundle.num_trials}, output_dir / "run_metadata.json")
    print(f"[run] Saved metrics to: {output_dir / 'metrics.csv'}")
    print(f"[run] Saved CV records to: {output_dir / 'cv_results.csv'}")

    if bool(cfg.get("experiment", {}).get("generate_report_after_run", False)):
        report_command(cfg)
