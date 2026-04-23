from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


@dataclass
class TrainParams:
    epochs: int = 50
    lr: float = 0.03
    batch_size: int = 256
    margin: float = 1.0
    max_iter: int = 300
    tol: float = 1e-2
    solver: str = "saga"


class SklearnLogisticClassifier:
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        C: float = 1.0,
        train_params: TrainParams | None = None,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.C = float(C)
        self.train_params = train_params or TrainParams()
        self.device = device
        self.seed = int(seed)
        self.model: LogisticRegression | None = None

    def fit(self, X, y):
        self.model = LogisticRegression(
            C=self.C,
            solver=self.train_params.solver,
            max_iter=max(50, int(self.train_params.max_iter)),
            tol=float(self.train_params.tol),
            random_state=self.seed,
            n_jobs=None,
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model not fit.")
        return self.model.predict(X)


class SklearnLinearSVMClassifier:
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        C: float = 1.0,
        train_params: TrainParams | None = None,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.C = float(C)
        self.train_params = train_params or TrainParams(max_iter=3000, tol=1e-4, solver="linear_svc")
        self.device = device
        self.seed = int(seed)
        self.model: LinearSVC | None = None

    def fit(self, X, y):
        self.model = LinearSVC(
            C=self.C,
            loss="squared_hinge",
            multi_class="ovr",
            dual="auto",
            tol=float(self.train_params.tol),
            max_iter=max(1000, int(self.train_params.max_iter)),
            random_state=self.seed,
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model not fit.")
        return self.model.predict(X)


class SklearnKNNClassifier:
    def __init__(self, n_neighbors: int = 5, weights: str = "uniform", device: str = "cpu"):
        self.n_neighbors = int(n_neighbors)
        self.weights = str(weights)
        self.device = device
        self.model: KNeighborsClassifier | None = None

    def fit(self, X, y):
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric="euclidean",
            algorithm="auto",
            n_jobs=None,
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("kNN model not fit.")
        return self.model.predict(X)



def build_train_params(cfg: dict[str, Any] | None) -> TrainParams:
    cfg = cfg or {}
    return TrainParams(
        epochs=int(cfg.get("epochs", 50)),
        lr=float(cfg.get("lr", 0.03)),
        batch_size=int(cfg.get("batch_size", 256)),
        margin=float(cfg.get("margin", 1.0)),
        max_iter=int(cfg.get("max_iter", 300)),
        tol=float(cfg.get("tol", 1e-2)),
        solver=str(cfg.get("solver", "saga")),
    )



def create_model(model_name: str, input_dim: int, num_classes: int, hyperparams: dict[str, Any],
                 train_cfg: dict[str, Any] | None, device: str, seed: int):
    name = model_name.lower()
    if name == "logistic":
        return SklearnLogisticClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            C=float(hyperparams.get("C", 1.0)),
            train_params=build_train_params(train_cfg),
            device=device,
            seed=seed,
        )
    if name == "linear_svm":
        return SklearnLinearSVMClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            C=float(hyperparams.get("C", 1.0)),
            train_params=build_train_params(train_cfg),
            device=device,
            seed=seed,
        )
    if name == "knn":
        return SklearnKNNClassifier(
            n_neighbors=int(hyperparams.get("n_neighbors", 5)),
            weights=str(hyperparams.get("weights", "uniform")),
            device=device,
        )
    raise ValueError(f"Unknown model name: {model_name}")
