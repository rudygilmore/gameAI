"""Load and validate `config.yaml` for training and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from env.chase_env import _validate_grid_side


def load_config(path: str | Path) -> dict[str, Any]:
    """Parse a YAML config file (e.g. **`chase/config.yaml`**)."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping, got {type(raw)}")
    return raw


def resolve_training_device(spec: str) -> torch.device:
    """Map **`training.device`** (`auto` / `cpu` / `cuda`) to a **torch.device**."""
    s = spec.lower().strip()
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def validate_config(config: dict[str, Any]) -> None:
    """Raise if **env.n** violates Chase grid rules (**N = 2^k**, **k ≥ 3**, **512 % N == 0**)."""
    env_cfg = config.get("env")
    if not isinstance(env_cfg, dict) or "n" not in env_cfg:
        raise ValueError("config must contain env.n")
    _validate_grid_side(int(env_cfg["n"]))
