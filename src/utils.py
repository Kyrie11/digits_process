from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    threads = int(os.environ.get("TORCH_NUM_THREADS", "1"))
    torch.set_num_threads(max(1, threads))


def get_device(requested: str | None = "auto") -> str:
    if requested is None or requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Any, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_serializable(data), f, sort_keys=False, allow_unicode=True)


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def save_json(data: Any, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(data), f, indent=2, ensure_ascii=False)


def slugify(*parts: Any) -> str:
    merged = "__".join("none" if p is None else str(p) for p in parts)
    safe = []
    for ch in merged:
        if ch.isalnum() or ch in {"-", "_", "."}:
            safe.append(ch)
        else:
            safe.append("-")
    return "".join(safe)


def resolve_output_dir(cfg: dict[str, Any]) -> Path:
    exp = cfg.setdefault("experiment", {})
    return ensure_dir(exp.get("output_dir", "outputs/default_run"))
