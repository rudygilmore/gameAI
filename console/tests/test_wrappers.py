"""Tests for ``src.wrappers`` (no ROMs needed; uses a fake env)."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from src.wrappers import Discretizer, RewardWrapper, VisionWrapper
from src.wrappers.discretizer import default_combos
from src.wrappers.vision import resize_frame, rgb_to_gray


# ---------------------------------------------------------------------------
# A minimal fake stable-retro-like env so we don't need ROMs.
# ---------------------------------------------------------------------------


class FakeRetroEnv(gym.Env):
    """A toy env mimicking stable-retro: ``MultiBinary`` actions + RGB frames."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        height: int = 32,
        width: int = 48,
        num_buttons: int = 12,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.MultiBinary(num_buttons)
        self.buttons = list(
            ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        )[:num_buttons]
        self._rng = np.random.default_rng(seed)
        self._steps = 0
        self._height = height
        self._width = width

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._steps = 0
        return self._frame(), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_arr = np.asarray(action, dtype=np.int8)
        assert action_arr.shape == (len(self.buttons),)
        self._steps += 1
        terminated = self._steps >= 25
        return self._frame(), float(action_arr.sum()), terminated, False, {"steps": self._steps}

    def _frame(self) -> np.ndarray:
        return self._rng.integers(0, 256, size=(self._height, self._width, 3), dtype=np.uint8)

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass


# ---------------------------------------------------------------------------
# Discretizer
# ---------------------------------------------------------------------------


def test_default_combos_is_noop_plus_singletons():
    env = FakeRetroEnv()
    combos = default_combos(env.buttons)
    assert combos[0] == []
    assert len(combos) == 1 + len(env.buttons)
    for combo, button in zip(combos[1:], env.buttons):
        assert combo == [button]


def test_discretizer_exposes_discrete_action_space():
    env = Discretizer(FakeRetroEnv())
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 13


def test_discretizer_action_table_one_hot_per_singleton():
    inner = FakeRetroEnv()
    env = Discretizer(inner)
    assert tuple(env.action(0)) == (0,) * 12
    for i, button in enumerate(inner.buttons, start=1):
        a = env.action(i)
        assert a.sum() == 1
        assert a[inner.buttons.index(button)] == 1


def test_discretizer_step_dispatches_correct_buttons():
    env = Discretizer(FakeRetroEnv())
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(1)
    # Action index 1 is a single-button press, so the FakeRetroEnv reports
    # reward == 1 (sum of pressed buttons).
    assert reward == 1.0
    assert obs.shape == (32, 48, 3)
    assert terminated is False


def test_discretizer_rejects_unknown_button():
    with pytest.raises(ValueError, match="unknown buttons"):
        Discretizer(FakeRetroEnv(), combos=[[], ["NOPE"]])


def test_discretizer_rejects_out_of_range_action():
    env = Discretizer(FakeRetroEnv())
    env.reset(seed=0)
    with pytest.raises(IndexError):
        env.step(99)


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------


def test_rgb_to_gray_uses_bt601_weights():
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    frame[0, 0] = (255, 0, 0)
    assert rgb_to_gray(frame)[0, 0] == int(round(255 * 0.299))


def test_resize_frame_shape():
    src = np.arange(64, dtype=np.uint8).reshape(8, 8)
    out = resize_frame(src, (4, 4))
    assert out.shape == (4, 4) and out.dtype == np.uint8


def test_vision_wrapper_initial_frame_repeats_to_fill_stack():
    env = VisionWrapper(FakeRetroEnv(), frame_stack=4)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (4, 32, 48)
    # All frames in a fresh stack should be identical (repeated initial frame).
    for k in range(1, 4):
        assert np.array_equal(obs[0], obs[k])


def test_vision_wrapper_advances_stack_on_step():
    inner = FakeRetroEnv(seed=42)
    env = VisionWrapper(inner, frame_stack=4)
    obs0, _ = env.reset(seed=42)
    obs1, _r, _t, _tr, _i = env.step(np.zeros(12, dtype=np.int8))
    # Newest frame is at index 3, prior at index 2 (=== first in obs0).
    assert np.array_equal(obs1[2], obs0[3])
    # The newest frame is different from the previous (random env).
    assert not np.array_equal(obs1[3], obs0[3])


def test_vision_wrapper_target_res_overrides_native_shape():
    env = VisionWrapper(FakeRetroEnv(), frame_stack=2, target_res=(16, 16))
    obs, _ = env.reset(seed=0)
    assert obs.shape == (2, 16, 16)


def test_vision_wrapper_rejects_non_rgb():
    class Mono(FakeRetroEnv):
        def __init__(self):
            super().__init__()
            self.observation_space = gym.spaces.Box(0, 255, shape=(32, 48), dtype=np.uint8)

    with pytest.raises(ValueError):
        VisionWrapper(Mono(), frame_stack=2)


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


def test_reward_wrapper_subclass_can_scale():
    class Scale(RewardWrapper):
        def reward(self, reward: float) -> float:
            return reward * 0.1

    env = Scale(Discretizer(FakeRetroEnv()))
    env.reset(seed=0)
    _obs, reward, _t, _tr, _i = env.step(1)
    # Action 1 presses a single button (raw reward 1.0); scaled to 0.1.
    assert reward == pytest.approx(0.1)
