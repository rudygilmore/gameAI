"""Unit tests for ``src.config``: TOML schema, defaults, and error messages."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from src.config import ConfigError, load_run_config, parse_run_config


def _full_training_block() -> dict:
    return {
        "seed": 0,
        "device": "cpu",
        "total_steps": 100,
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "batch_size": 4,
        "replay_capacity": 32,
        "target_update_period": 10,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 50,
    }


def test_parse_minimal_required_keys_uses_defaults():
    cfg = parse_run_config(
        {
            "game": "Airstriker-Genesis",
            "training": _full_training_block(),
        }
    )
    assert cfg.game == "Airstriker-Genesis"
    assert cfg.state is None
    assert cfg.double is False
    assert cfg.dueling is False
    assert cfg.frame_stack == 4
    assert cfg.reward_shaping is False
    assert cfg.reward_module is None
    assert cfg.target_res is None
    assert cfg.training.total_steps == 100


def test_parse_overrides_defaults_when_provided():
    cfg = parse_run_config(
        {
            "game": "MsPacMan-Nes-v0",
            "state": "1Player.Level1",
            "double": True,
            "dueling": True,
            "frame_stack": 2,
            "target_res": [84, 84],
            "training": _full_training_block(),
        }
    )
    assert cfg.state == "1Player.Level1"
    assert cfg.double is True
    assert cfg.dueling is True
    assert cfg.frame_stack == 2
    assert cfg.target_res == (84, 84)


def test_parse_missing_game_raises():
    with pytest.raises(ConfigError, match="missing required key 'game'"):
        parse_run_config({"training": _full_training_block()})


def test_parse_missing_training_table_raises():
    with pytest.raises(ConfigError, match="missing required key 'training'"):
        parse_run_config({"game": "Airstriker-Genesis"})


@pytest.mark.parametrize(
    "key",
    [
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
    ],
)
def test_parse_missing_training_key_names_the_key(key):
    block = _full_training_block()
    block.pop(key)
    with pytest.raises(ConfigError, match=f"missing required key '{key}' in \\[training\\]"):
        parse_run_config({"game": "Airstriker-Genesis", "training": block})


def test_reward_shaping_without_module_raises():
    with pytest.raises(ConfigError, match="requires a 'reward_module'"):
        parse_run_config(
            {
                "game": "Airstriker-Genesis",
                "reward_shaping": True,
                "training": _full_training_block(),
            }
        )


def test_target_res_must_be_pair_of_positive_ints():
    base = {"game": "Airstriker-Genesis", "training": _full_training_block()}
    with pytest.raises(ConfigError, match="target_res"):
        parse_run_config({**base, "target_res": [84]})
    with pytest.raises(ConfigError, match="target_res"):
        parse_run_config({**base, "target_res": [84, "wide"]})
    with pytest.raises(ConfigError, match="target_res"):
        parse_run_config({**base, "target_res": [0, 84]})


def test_load_run_config_reads_example_toml():
    example = (
        Path(__file__).resolve().parents[1] / "configs" / "example_run.toml"
    )
    cfg = load_run_config(example)
    assert cfg.game == "Airstriker-Genesis"
    assert cfg.state == "Level1"
    assert cfg.training.total_steps == 1_000_000
    assert cfg.training.epsilon_end == pytest.approx(0.05)
    assert cfg.source_path is not None and cfg.source_path.is_file()


def test_load_run_config_reads_handcrafted_toml(tmp_path):
    toml = tmp_path / "run.toml"
    toml.write_text(
        textwrap.dedent(
            """
            game = "Airstriker-Genesis"
            double = true
            dueling = true
            frame_stack = 2

            [training]
            seed = 7
            device = "cpu"
            total_steps = 50
            learning_rate = 0.0005
            gamma = 0.95
            batch_size = 8
            replay_capacity = 64
            target_update_period = 5
            epsilon_start = 1.0
            epsilon_end = 0.1
            epsilon_decay_steps = 20
            """
        ).strip()
    )
    cfg = load_run_config(toml)
    assert cfg.double is True and cfg.dueling is True
    assert cfg.training.seed == 7
    assert cfg.training.learning_rate == pytest.approx(0.0005)
