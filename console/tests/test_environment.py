"""Environment-factory tests.

The full ``make_env`` path requires a real ``stable-retro`` install with at
least one imported ROM. Per Milestone B in EXECPLAN_UNIVERSAL_RETRO_DQN.md,
this test is gated on:

* the ``stable_retro`` Python module being importable, and
* the env var ``UNIVERSAL_RETRO_DQN_TEST_GAME`` naming an imported game id
  (e.g. ``Airstriker-Genesis-v0``).

When either condition is missing the test ``skip``s rather than fails so
``pytest -q`` stays green on machines without ROMs (matches the "Validation
and Acceptance" section's minimum bar).

The reward-wrapper loader is tested with a fake env (no ROMs needed).
"""

from __future__ import annotations

import importlib.util
import os
import textwrap
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from src.config import RunConfig, TrainingConfig, parse_run_config
from src.environment import _load_reward_wrapper_class, make_env
from src.wrappers.reward_wrapper import RewardWrapper


def _training() -> TrainingConfig:
    return TrainingConfig(
        seed=0,
        device="cpu",
        total_steps=4,
        learning_rate=1e-4,
        gamma=0.99,
        batch_size=2,
        replay_capacity=8,
        target_update_period=4,
        epsilon_start=1.0,
        epsilon_end=1.0,
        epsilon_decay_steps=4,
    )


def _env_var_game() -> str | None:
    return os.environ.get("UNIVERSAL_RETRO_DQN_TEST_GAME") or None


@pytest.mark.skipif(
    importlib.util.find_spec("stable_retro") is None,
    reason="stable_retro not installed",
)
def test_make_env_builds_full_stack_when_game_available():
    game = _env_var_game()
    if game is None:
        pytest.skip(
            "set UNIVERSAL_RETRO_DQN_TEST_GAME=<imported_game_id> to exercise this test"
        )
    cfg = parse_run_config(
        {
            "game": game,
            "frame_stack": 2,
            "training": {
                "seed": 0,
                "device": "cpu",
                "total_steps": 4,
                "learning_rate": 1e-4,
                "gamma": 0.99,
                "batch_size": 2,
                "replay_capacity": 8,
                "target_update_period": 4,
                "epsilon_start": 1.0,
                "epsilon_end": 1.0,
                "epsilon_decay_steps": 4,
            },
        }
    )
    env = make_env(cfg)
    try:
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n >= 2
        assert env.observation_space.shape is not None
        assert env.observation_space.shape[0] == 2  # frame_stack
        obs, _info = env.reset(seed=0)
        assert obs.dtype == np.uint8
        assert obs.shape == env.observation_space.shape
        next_obs, reward, terminated, truncated, _ = env.step(0)
        assert next_obs.shape == obs.shape
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
    finally:
        env.close()


def test_load_reward_wrapper_class_finds_module(tmp_path: Path):
    # Lay out a minimal mock console tree so _rewards_dir() resolves correctly.
    cfg_dir = tmp_path / "configs"
    rewards_dir = cfg_dir / "rewards"
    rewards_dir.mkdir(parents=True)
    (rewards_dir / "demo.py").write_text(
        textwrap.dedent(
            """
            from src.wrappers.reward_wrapper import RewardWrapper

            class Wrapper(RewardWrapper):
                def reward(self, reward):
                    return reward + 1.0
            """
        ).strip()
    )
    toml = cfg_dir / "demo.toml"
    toml.write_text("game = 'X'\n[training]\nseed=0\ndevice='cpu'\ntotal_steps=1\n"
                    "learning_rate=1e-4\ngamma=0.99\nbatch_size=1\nreplay_capacity=1\n"
                    "target_update_period=1\nepsilon_start=1.0\nepsilon_end=1.0\n"
                    "epsilon_decay_steps=1\n")
    cfg = RunConfig(
        game="X",
        state=None,
        double=False,
        dueling=False,
        frame_stack=4,
        reward_shaping=True,
        reward_module="demo",
        target_res=None,
        training=_training(),
        source_path=toml.resolve(),
    )
    cls = _load_reward_wrapper_class(cfg)
    assert issubclass(cls, RewardWrapper)


def test_load_reward_wrapper_class_missing_file_raises(tmp_path: Path):
    cfg_dir = tmp_path / "configs"
    (cfg_dir / "rewards").mkdir(parents=True)
    toml = cfg_dir / "demo.toml"
    toml.write_text("ignored")
    cfg = RunConfig(
        game="X",
        state=None,
        double=False,
        dueling=False,
        frame_stack=4,
        reward_shaping=True,
        reward_module="missing",
        target_res=None,
        training=_training(),
        source_path=toml.resolve(),
    )
    with pytest.raises(FileNotFoundError):
        _load_reward_wrapper_class(cfg)


def test_load_reward_wrapper_class_wrong_class_raises(tmp_path: Path):
    cfg_dir = tmp_path / "configs"
    rewards_dir = cfg_dir / "rewards"
    rewards_dir.mkdir(parents=True)
    (rewards_dir / "bad.py").write_text("class NotWrapper: pass\n")
    toml = cfg_dir / "demo.toml"
    toml.write_text("ignored")
    cfg = RunConfig(
        game="X",
        state=None,
        double=False,
        dueling=False,
        frame_stack=4,
        reward_shaping=True,
        reward_module="bad",
        target_res=None,
        training=_training(),
        source_path=toml.resolve(),
    )
    with pytest.raises(AttributeError):
        _load_reward_wrapper_class(cfg)


def test_load_reward_wrapper_class_non_subclass_raises(tmp_path: Path):
    cfg_dir = tmp_path / "configs"
    rewards_dir = cfg_dir / "rewards"
    rewards_dir.mkdir(parents=True)
    (rewards_dir / "wrong.py").write_text("class Wrapper: pass\n")
    toml = cfg_dir / "demo.toml"
    toml.write_text("ignored")
    cfg = RunConfig(
        game="X",
        state=None,
        double=False,
        dueling=False,
        frame_stack=4,
        reward_shaping=True,
        reward_module="wrong",
        target_res=None,
        training=_training(),
        source_path=toml.resolve(),
    )
    with pytest.raises(TypeError):
        _load_reward_wrapper_class(cfg)
