from __future__ import annotations

from pathlib import Path
from typing import Any

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import  numpy as np

from .data import get_trial_data, load_digits_mat
from .metrics import build_confusion_matrix
from .preprocess import add_gaussian_noise
from .utils import ensure_dir, resolve_output_dir, save_json


def _format_mean_std(mean_value: float, std_value: float | None) -> str:
    if std_value is None or np.isnan(std_value):
        std_value = 0.0
    return f"{mean_value:.4f} ± {std_value:.4f}"


def _config_key(df):
    pca_str = df["pca_dim"].apply(lambda v: "none" if pd.isna(v) else str(int(v)))
    return df["preprocess"].astype(str) + "|" + df["model"].astype(str) + "|" + pca_str


def _choose_best_by_group(df, group_cols, score_col):
    ordered = df.sort_values([score_col, "std_test_accuracy"], ascending=[False, True])
    return ordered.drop_duplicates(subset=group_cols).reset_index(drop=True)


def _save_robustness_curves(metrics_df: pd.DataFrame, clean_summary_best: pd.DataFrame, figure_path: Path) -> None:
    metrics_df = metrics_df.copy()
    clean_summary_best = clean_summary_best.copy()

    metrics_df["config_key"] = _config_key(metrics_df)
    clean_summary_best["config_key"] = _config_key(clean_summary_best)

    target_configs = clean_summary_best[["config_key", "model", "preprocess", "pca_dim"]].drop_duplicates()
    merged = metrics_df.merge(
        target_configs[["config_key"]].drop_duplicates(),
        on="config_key",
        how="inner",
    )

    if merged.empty:
        return

    summary = (
        merged.groupby(
            ["model", "preprocess", "pca_dim", "noise_sigma"],
            dropna=False,
            as_index=False,
        )["test_accuracy"]
        .mean()
        .sort_values(["model", "noise_sigma"])
    )

    plt.figure(figsize=(10, 6))

    for _, group in summary.groupby(["model", "preprocess", "pca_dim"], dropna=False):
        pca_value = group.iloc[0]["pca_dim"]
        pca_display = "none" if pd.isna(pca_value) else int(pca_value)
        label = f"{group.iloc[0]['model']} | {group.iloc[0]['preprocess']} | pca={pca_display}"
        plt.plot(group["noise_sigma"], group["test_accuracy"], marker="o", label=label)

    plt.xlabel("Gaussian noise sigma")
    plt.ylabel("Mean accuracy")
    plt.title("Robustness curves of the best clean configuration for each model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=220)
    plt.close()


def _save_robustness_curves(metrics_df: pd.DataFrame, clean_summary_best: pd.DataFrame, figure_path: Path) -> None:
    metrics_df = metrics_df.copy()
    clean_summary_best["config_key"] = _config_key(clean_summary_best)
    metrics_df["config_key"] = _config_key(metrics_df)
    clean_summary_best["config_key"] = _config_key(clean_summary_best)
    target_configs = clean_summary_best[["config_key", "model", "preprocess", "pca_dim"]].drop_duplicates()
    merged = metrics_df.merge(target_configs[["config_key"]].drop_duplicates(), on="config_key", how="inner")
    if merged.empty:
        return
    summary = (
        merged.groupby(["model", "preprocess", "pca_dim", "noise_sigma"], dropna=False, as_index=False)["test_accuracy"]
        .mean()
        .sort_values(["model", "noise_sigma"])
    )
    plt.figure(figsize=(10, 6))
    for _, group in summary.groupby(["model", "preprocess", "pca_dim"], dropna=False):
        pca_display = "none" if pd.isna(group.iloc[0]["pca_dim"]) else int(group.iloc[0]["pca_dim"])
        label = f"{group.iloc[0]['model']} | {group.iloc[0]['preprocess']} | pca={pca_display}"
        plt.plot(group["noise_sigma"], group["test_accuracy"], marker="o", label=label)
    plt.xlabel("Gaussian noise sigma")
    plt.ylabel("Mean accuracy")
    plt.title("Robustness curves of the best clean configuration for each model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=220)
    plt.close()


def _load_prediction_artifact(run_dir: Path, relative_path: str):
    return np.load(run_dir / relative_path, allow_pickle=True)


def _save_confusion_figure(cm: np.ndarray, class_names: list[str], title: str, out_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45)
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    threshold = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="white" if cm[i, j] > threshold else "black")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _save_failure_cases(
    bundle,
    artifact,
    trial: int,
    pred_key: str,
    out_path: Path,
    title: str,
    top_k: int = 5,
    images=None,
):
    _, _, X_test, y_test, _, _ = get_trial_data(bundle, int(trial))
    X_vis = X_test if images is None else images

    y_true = artifact["y_true"]
    y_pred = artifact[pred_key]
    assert np.array_equal(y_true, y_test)

    correct_idx = np.where(y_true == y_pred)[0][:top_k]
    wrong_idx = np.where(y_true != y_pred)[0][:top_k]

    side = int(np.sqrt(bundle.num_features))
    cols = max(len(correct_idx), len(wrong_idx), 1)
    rows = 2
    plt.figure(figsize=(cols * 2.2, rows * 2.5))

    for i in range(cols):
        plt.subplot(rows, cols, i + 1)
        if i < len(correct_idx):
            idx = int(correct_idx[i])
            plt.imshow(X_vis[idx].reshape(side, side), cmap="gray")
            plt.title(
                f"Correct\nT={bundle.class_names[y_true[idx]]}, P={bundle.class_names[y_pred[idx]]}"
            )
        plt.axis("off")

    for i in range(cols):
        plt.subplot(rows, cols, cols + i + 1)
        if i < len(wrong_idx):
            idx = int(wrong_idx[i])
            plt.imshow(X_vis[idx].reshape(side, side), cmap="gray")
            plt.title(
                f"Wrong\nT={bundle.class_names[y_true[idx]]}, P={bundle.class_names[y_pred[idx]]}"
            )
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=220)
    plt.close()

def _save_accuracy_vs_pca(clean_df: pd.DataFrame, figure_path: Path) -> None:
    pca_df = clean_df[clean_df["preprocess"].isin(["pca", "standard_pca"])].copy()
    if pca_df.empty:
        return

    summary = (
        pca_df.groupby(["preprocess", "model", "pca_dim"], as_index=False)["test_accuracy"]
        .mean()
        .sort_values(["preprocess", "model", "pca_dim"])
    )

    plt.figure(figsize=(10, 6))
    for (preprocess, model), group in summary.groupby(["preprocess", "model"]):
        plt.plot(
            group["pca_dim"],
            group["test_accuracy"],
            marker="o",
            label=f"{preprocess}-{model}",
        )

    plt.xlabel("PCA dimension")
    plt.ylabel("Mean clean accuracy")
    plt.title("Accuracy vs PCA dimension")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=220)
    plt.close()

def report_command(cfg: dict[str, Any]) -> None:
    run_dir = resolve_output_dir(cfg)
    tables_dir = ensure_dir(run_dir / "tables")
    figures_dir = ensure_dir(run_dir / "figures")

    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.csv not found: {metrics_path}")

    metrics_df = pd.read_csv(metrics_path)
    metrics_df["config_key"] = _config_key(metrics_df)

    clean_df = metrics_df[metrics_df["noise_sigma"] == 0.0].copy()
    clean_summary = (
        clean_df.groupby(["config_key", "preprocess", "model", "pca_dim"], dropna=False, as_index=False)
        .agg(
            mean_test_accuracy=("test_accuracy", "mean"),
            std_test_accuracy=("test_accuracy", "std"),
            mean_cv_accuracy=("cv_mean_accuracy", "mean"),
            feature_dim_after_transform=("feature_dim_after_transform", "first"),
        )
        .sort_values("mean_test_accuracy", ascending=False)
    )
    clean_summary.to_csv(tables_dir / "clean_summary_by_config.csv", index=False)

    best_clean_per_method_model = _choose_best_by_group(clean_summary, ["preprocess", "model"], "mean_test_accuracy")
    best_clean_per_method_model["score"] = [
        _format_mean_std(m, s) for m, s in zip(best_clean_per_method_model["mean_test_accuracy"], best_clean_per_method_model["std_test_accuracy"])
    ]
    best_clean_per_method_model.to_csv(tables_dir / "main_accuracy_table_long.csv", index=False)

    main_table = best_clean_per_method_model.pivot(index="preprocess", columns="model", values="score")
    main_table.to_csv(tables_dir / "main_accuracy_table.csv")
    with open(tables_dir / "main_accuracy_table.tex", "w", encoding="utf-8") as f:
        f.write(main_table.fillna("-").to_latex())

    noise_summary = (
        metrics_df.groupby(["config_key", "preprocess", "model", "pca_dim", "noise_sigma"], dropna=False, as_index=False)
        .agg(mean_test_accuracy=("test_accuracy", "mean"), std_test_accuracy=("test_accuracy", "std"))
    )
    noise_summary.to_csv(tables_dir / "noise_summary_by_config.csv", index=False)

    merged_noise = noise_summary.merge(
        clean_summary[["config_key", "mean_test_accuracy"]].rename(columns={"mean_test_accuracy": "clean_mean_accuracy"}),
        on="config_key",
        how="left",
    )
    merged_noise["robustness_drop"] = merged_noise["clean_mean_accuracy"] - merged_noise["mean_test_accuracy"]
    merged_noise.to_csv(tables_dir / "robustness_drop_table.csv", index=False)

    best_clean_overall = clean_summary.sort_values("mean_test_accuracy", ascending=False).iloc[0].to_dict()
    highest_noise = float(cfg.get("report", {}).get("highest_noise_for_robustness", metrics_df["noise_sigma"].max()))
    robust_summary = merged_noise[np.isclose(merged_noise["noise_sigma"], highest_noise)].copy()
    if not robust_summary.empty:
        robust_summary = robust_summary.sort_values("mean_test_accuracy", ascending=False)
        robust_summary.to_csv(tables_dir / "robustness_highest_noise_summary.csv", index=False)
        with open(tables_dir / "robustness_highest_noise_summary.tex", "w", encoding="utf-8") as f:
            f.write(robust_summary.to_latex(index=False))
    robust_candidates = noise_summary[np.isclose(noise_summary["noise_sigma"], highest_noise)]
    if robust_candidates.empty:
        robust_candidates = noise_summary[noise_summary["noise_sigma"] == noise_summary["noise_sigma"].max()]

    robust_candidates = merged_noise[np.isclose(merged_noise["noise_sigma"], highest_noise)].copy()
    best_robust_overall = robust_candidates.sort_values(
        ["robustness_drop", "mean_test_accuracy"],
        ascending=[True, False]
    ).iloc[0].to_dict()

    best_clean_per_model = _choose_best_by_group(clean_summary, ["model"], "mean_test_accuracy")

    best_configs = {
        "best_clean_overall": best_clean_overall,
        "best_robust_overall": best_robust_overall,
        "best_clean_per_model": best_clean_per_model.to_dict(orient="records"),
    }
    save_json(best_configs, run_dir / "best_configs.json")

    accuracy_vs_pca_path = figures_dir / "accuracy_vs_pca.png"
    dimension_curve_path = figures_dir / "dimension_accuracy_curve.png"
    robustness_best_path = figures_dir / "robustness_curves_best_per_model.png"
    robustness_curve_path = figures_dir / "robustness_curve.png"

    _save_accuracy_vs_pca(clean_df, accuracy_vs_pca_path)
    if accuracy_vs_pca_path.exists():
        import shutil
        shutil.copyfile(accuracy_vs_pca_path, dimension_curve_path)

    _save_robustness_curves(metrics_df, best_clean_per_model, robustness_best_path)
    if robustness_best_path.exists():
        import shutil
        shutil.copyfile(robustness_best_path, robustness_curve_path)

    bundle = load_digits_mat(cfg["data"]["mat_path"], auto_normalize_01=bool(cfg["data"].get("auto_normalize_01", True)))

    best_clean_rows = clean_df[clean_df["config_key"] == best_clean_overall["config_key"]]
    best_clean_row = best_clean_rows.sort_values("test_accuracy", ascending=False).iloc[0]
    clean_artifact = _load_prediction_artifact(run_dir, best_clean_row["prediction_file"])
    clean_cm = build_confusion_matrix(clean_artifact["y_true"], clean_artifact["y_pred_clean"], labels=list(range(bundle.num_classes)))
    _save_confusion_figure(clean_cm, bundle.class_names, "Confusion Matrix - Best Clean Model", figures_dir / "confusion_best_clean.png")
    _save_failure_cases(
        bundle=bundle,
        artifact=clean_artifact,
        trial=int(best_clean_row["trial"]),
        pred_key="y_pred_clean",
        out_path=figures_dir / "failure_cases_best_clean.png",
        title="Success and failure cases - best clean model",
        top_k=int(cfg.get("report", {}).get("top_k_cases", 5)),
    )

    robust_rows = metrics_df[
        (metrics_df["config_key"] == best_robust_overall["config_key"]) &
        np.isclose(metrics_df["noise_sigma"], float(best_robust_overall["noise_sigma"]))
    ].copy()
    best_robust_row = robust_rows.sort_values("test_accuracy", ascending=False).iloc[0]
    robust_artifact = _load_prediction_artifact(run_dir, best_robust_row["prediction_file"])
    noise_sigma = float(best_robust_overall["noise_sigma"])

    noise_levels = np.asarray(robust_artifact["noise_levels"], dtype=float)
    matches = np.where(np.isclose(noise_levels, noise_sigma))[0]
    if len(matches) > 0:
        robust_pred = robust_artifact["noisy_preds"][int(matches[0])]
    else:
        robust_pred = robust_artifact["y_pred_clean"]

    # 重新取出该 trial 的 clean test images，并按和 runner.py 一致的方式加噪
    _, _, X_test_raw, _, _, _ = get_trial_data(bundle, int(best_robust_row["trial"]))

    if noise_sigma > 0:
        robust_images = add_gaussian_noise(
            X_test_raw,
            sigma=noise_sigma,
            seed=int(cfg["experiment"].get("seed", 42)) + int(best_robust_row["trial"]) * 1000 + int(100 * noise_sigma),
            clip_min=0.0,
            clip_max=1.0,
        )
    else:
        robust_images = X_test_raw

    robust_cm = build_confusion_matrix(
        robust_artifact["y_true"],
        robust_pred,
        labels=list(range(bundle.num_classes)),
    )
    _save_confusion_figure(
        robust_cm,
        bundle.class_names,
        "Confusion Matrix - Best Robust Model",
        figures_dir / "confusion_best_robust.png",
    )

    _save_failure_cases(
        bundle=bundle,
        artifact={
            "y_true": robust_artifact["y_true"],
            "y_pred_selected": robust_pred,
        },
        trial=int(best_robust_row["trial"]),
        pred_key="y_pred_selected",
        out_path=figures_dir / "failure_cases_best_robust.png",
        title=f"Noisy test cases at sigma={noise_sigma} for the best robust model",
        top_k=int(cfg.get("report", {}).get("top_k_cases", 5)),
        images=robust_images,
    )

    print(f"[report] Saved tables to: {tables_dir}")
    print(f"[report] Saved figures to: {figures_dir}")
