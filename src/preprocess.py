from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from sklearn.decomposition import PCA


class BaseTransform:
    def fit(self, X: torch.Tensor):
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        return self.fit(X).transform(X)


class IdentityTransform(BaseTransform):
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return X


@dataclass
class StandardizeTransform(BaseTransform):
    eps: float = 1e-6
    mean: Optional[torch.Tensor] = None
    std: Optional[torch.Tensor] = None

    def fit(self, X: torch.Tensor):
        self.mean = X.mean(dim=0, keepdim=True)
        self.std = X.std(dim=0, unbiased=False, keepdim=True).clamp_min(self.eps)
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardizeTransform must be fit before transform.")
        return (X - self.mean) / self.std


@dataclass
class MinMaxTransform(BaseTransform):
    eps: float = 1e-6
    min_: Optional[torch.Tensor] = None
    range_: Optional[torch.Tensor] = None

    def fit(self, X: torch.Tensor):
        self.min_ = X.min(dim=0, keepdim=True).values
        max_ = X.max(dim=0, keepdim=True).values
        self.range_ = (max_ - self.min_).clamp_min(self.eps)
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.min_ is None or self.range_ is None:
            raise RuntimeError("MinMaxTransform must be fit before transform.")
        return (X - self.min_) / self.range_


@dataclass
class PCATransform(BaseTransform):
    n_components: int
    mean: Optional[torch.Tensor] = None
    pca_model: Optional[PCA] = None

    def fit(self, X: torch.Tensor):
        if self.n_components is None:
            raise ValueError("PCA requires n_components.")
        X_np = X.detach().cpu().numpy()
        max_components = min(X_np.shape[0], X_np.shape[1])
        k = min(int(self.n_components), max_components)
        self.mean = X.mean(dim=0, keepdim=True)
        solver = "randomized" if k < max_components else "full"
        self.pca_model = PCA(n_components=k, svd_solver=solver, random_state=0)
        self.pca_model.fit(X_np)
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.pca_model is None:
            raise RuntimeError("PCATransform must be fit before transform.")
        X_np = X.detach().cpu().numpy()
        transformed = self.pca_model.transform(X_np)
        return torch.as_tensor(transformed, dtype=torch.float32)


class SequentialTransform(BaseTransform):
    def __init__(self, transforms: list[BaseTransform]):
        self.transforms = transforms

    def fit(self, X: torch.Tensor):
        cur = X
        for transform in self.transforms:
            transform.fit(cur)
            cur = transform.transform(cur)
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        cur = X
        for transform in self.transforms:
            cur = transform.transform(cur)
        return cur


def build_preprocessor(method: str, n_components: int | None = None) -> SequentialTransform:
    method = method.lower()
    if method == "raw":
        return SequentialTransform([IdentityTransform()])
    if method == "standard":
        return SequentialTransform([StandardizeTransform()])
    if method == "minmax":
        return SequentialTransform([MinMaxTransform()])
    if method == "pca":
        if n_components is None:
            raise ValueError("Preprocess method 'pca' requires n_components.")
        return SequentialTransform([PCATransform(n_components=n_components)])
    if method == "standard_pca":
        if n_components is None:
            raise ValueError("Preprocess method 'standard_pca' requires n_components.")
        return SequentialTransform([StandardizeTransform(), PCATransform(n_components=n_components)])
    raise ValueError(f"Unknown preprocess method: {method}")


def add_gaussian_noise(X, sigma: float, seed: int | None = None, clip_min: float = 0.0, clip_max: float = 1.0):
    if sigma <= 0:
        return X.copy()
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(int(seed))
    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    noise = torch.randn(X_tensor.shape, generator=generator, dtype=torch.float32)
    noisy = torch.clamp(X_tensor + sigma * noise, min=clip_min, max=clip_max)
    return noisy.numpy()
