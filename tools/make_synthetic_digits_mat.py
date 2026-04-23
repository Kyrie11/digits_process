from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import savemat


def make_dataset(samples_per_class: int = 40, n_trials: int = 2, seed: int = 7):
    rng = np.random.default_rng(seed)
    num_classes = 10
    feature_dim = 28 * 28
    centers = rng.normal(0.0, 1.0, size=(num_classes, feature_dim)).astype(np.float32)

    X_list = []
    y_list = []
    for cls in range(num_classes):
        for _ in range(samples_per_class):
            sample = centers[cls] + 0.55 * rng.normal(size=feature_dim)
            X_list.append(sample)
            y_list.append(cls)

    X = np.stack(X_list).astype(np.float32)
    X = (X - X.min()) / (X.max() - X.min() + 1e-12)
    y = np.array(y_list, dtype=np.int64)

    n = len(y)
    train_rows = []
    test_rows = []
    for trial in range(n_trials):
        perm = rng.permutation(n)
        split = int(0.6 * n)
        train_rows.append(perm[:split] + 1)  # MATLAB-style 1-based indexing
        test_rows.append(perm[split:] + 1)

    trainset = np.stack(train_rows, axis=0)
    testset = np.stack(test_rows, axis=0)
    return X, y, trainset, testset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/synthetic_digits4000.mat")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X, y, trainset, testset = make_dataset()
    payload = {
        "X": X.T,  # mimic common MATLAB storage: 784 x N
        "Y": y.reshape(1, -1),
        "trainset": trainset,
        "testset": testset,
    }
    savemat(out_path, payload)
    print(f"Synthetic dataset saved to: {out_path}")


if __name__ == "__main__":
    main()
