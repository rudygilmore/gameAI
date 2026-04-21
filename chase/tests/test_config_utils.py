from __future__ import annotations

from pathlib import Path

import pytest

from src.config_utils import load_config, resolve_training_device, validate_config


def test_load_config_reads_yaml() -> None:
    root = Path(__file__).resolve().parent.parent
    cfg = load_config(root / "config.yaml")
    assert cfg["env"]["id"] == "Chase-v0"
    assert int(cfg["env"]["n"]) >= 8


def test_validate_config_accepts_default_n() -> None:
    root = Path(__file__).resolve().parent.parent
    cfg = load_config(root / "config.yaml")
    validate_config(cfg)


def test_validate_config_rejects_invalid_grid_n() -> None:
    with pytest.raises(ValueError, match="power of 2"):
        validate_config({"env": {"n": 12}})


def test_resolve_training_device_cpu() -> None:
    assert resolve_training_device("cpu").type == "cpu"
