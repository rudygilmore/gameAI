"""Run-config loader for the Universal Retro-DQN.

A single TOML file describes one training or play session. The schema, defaults,
and a worked example are specified in
``console/EXECPLAN_UNIVERSAL_RETRO_DQN.md`` (see "Example run TOML"). This
module is the only place that knows how to parse that file; all downstream
consumers receive the typed :class:`RunConfig` dataclass below.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

if sys.version_info >= (3, 11):
    import tomllib as _tomllib
else:
    import tomli as _tomllib


_DEFAULT_FRAME_STACK = 4
_TRAINING_REQUIRED_KEYS: tuple[str, ...] = (
    "seed",
    "device",
    "total_steps",
    "learning_rate",
    "gamma",
    "batch_size",
    "replay_capacity",
    "target_update_period",
    "epsilon_start",
    "epsilon_end",
    "epsilon_decay_steps",
)


class ConfigError(ValueError):
    """Raised when a run TOML is missing a required key or has an invalid type."""


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters consumed by ``src.train`` and ``src.agent``."""

    seed: int
    device: str
    total_steps: int
    learning_rate: float
    gamma: float
    batch_size: int
    replay_capacity: int
    target_update_period: int
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int


@dataclass(frozen=True)
class RunConfig:
    """Parsed contents of a single run TOML file."""

    game: str
    state: str | None
    double: bool
    dueling: bool
    frame_stack: int
    reward_shaping: bool
    reward_module: str | None
    target_res: tuple[int, int] | None
    training: TrainingConfig
    source_path: Path | None = field(default=None, compare=False)


def _require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"missing required key '{key}' in {where}")
    return mapping[key]


def _typed(value: Any, expected: type | tuple[type, ...], key: str, where: str) -> Any:
    if not isinstance(value, expected):
        names = (
            expected.__name__
            if isinstance(expected, type)
            else ", ".join(t.__name__ for t in expected)
        )
        raise ConfigError(
            f"key '{key}' in {where} must be {names}, got {type(value).__name__}"
        )
    return value


def _parse_target_res(raw: Any) -> tuple[int, int]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ConfigError("'target_res' must be a [height, width] pair")
    h, w = raw
    if not isinstance(h, int) or not isinstance(w, int):
        raise ConfigError("'target_res' entries must both be integers")
    if h <= 0 or w <= 0:
        raise ConfigError("'target_res' entries must both be positive")
    return (h, w)


def _parse_training(raw: Any) -> TrainingConfig:
    if not isinstance(raw, dict):
        raise ConfigError("[training] table is missing or not a table")
    where = "[training]"
    for key in _TRAINING_REQUIRED_KEYS:
        if key not in raw:
            raise ConfigError(f"missing required key '{key}' in {where}")
    return TrainingConfig(
        seed=int(_typed(raw["seed"], int, "seed", where)),
        device=str(_typed(raw["device"], str, "device", where)),
        total_steps=int(_typed(raw["total_steps"], int, "total_steps", where)),
        learning_rate=float(_typed(raw["learning_rate"], (int, float), "learning_rate", where)),
        gamma=float(_typed(raw["gamma"], (int, float), "gamma", where)),
        batch_size=int(_typed(raw["batch_size"], int, "batch_size", where)),
        replay_capacity=int(_typed(raw["replay_capacity"], int, "replay_capacity", where)),
        target_update_period=int(
            _typed(raw["target_update_period"], int, "target_update_period", where)
        ),
        epsilon_start=float(_typed(raw["epsilon_start"], (int, float), "epsilon_start", where)),
        epsilon_end=float(_typed(raw["epsilon_end"], (int, float), "epsilon_end", where)),
        epsilon_decay_steps=int(
            _typed(raw["epsilon_decay_steps"], int, "epsilon_decay_steps", where)
        ),
    )


def parse_run_config(raw: Mapping[str, Any], *, source_path: Path | None = None) -> RunConfig:
    """Validate a previously-loaded TOML mapping and return a :class:`RunConfig`."""

    where = "<run config>"
    game = str(_typed(_require(raw, "game", where), str, "game", where))
    state_raw = raw.get("state")
    if state_raw is not None:
        state = str(_typed(state_raw, str, "state", where))
    else:
        state = None

    double = bool(_typed(raw.get("double", False), bool, "double", where))
    dueling = bool(_typed(raw.get("dueling", False), bool, "dueling", where))

    frame_stack_val = raw.get("frame_stack", _DEFAULT_FRAME_STACK)
    frame_stack = int(_typed(frame_stack_val, int, "frame_stack", where))
    if frame_stack < 1:
        raise ConfigError("'frame_stack' must be >= 1")

    reward_shaping = bool(_typed(raw.get("reward_shaping", False), bool, "reward_shaping", where))
    reward_module_raw = raw.get("reward_module")
    if reward_module_raw is not None:
        reward_module = str(_typed(reward_module_raw, str, "reward_module", where))
    else:
        reward_module = None

    if reward_shaping and reward_module is None:
        raise ConfigError(
            "'reward_shaping = true' requires a 'reward_module' string naming a "
            "file under configs/rewards/"
        )

    target_res = _parse_target_res(raw["target_res"]) if "target_res" in raw else None

    training = _parse_training(_require(raw, "training", where))

    return RunConfig(
        game=game,
        state=state,
        double=double,
        dueling=dueling,
        frame_stack=frame_stack,
        reward_shaping=reward_shaping,
        reward_module=reward_module,
        target_res=target_res,
        training=training,
        source_path=source_path,
    )


def load_run_config(path: str | Path) -> RunConfig:
    """Read a TOML file from ``path`` and return the validated :class:`RunConfig`."""

    p = Path(path)
    with p.open("rb") as f:
        data = _tomllib.load(f)
    return parse_run_config(data, source_path=p.resolve())
