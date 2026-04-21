"""Observation wrappers for Chase RL training."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torchvision.transforms.functional import InterpolationMode, resize


class ChaseObservationWrapper(gym.ObservationWrapper):
    """Convert Chase RGB observations to normalized CHW float32 tensors."""

    def __init__(self, env: gym.Env, resize_size: int = 64) -> None:
        super().__init__(env)
        if resize_size <= 0:
            raise ValueError(f"resize_size must be positive, got {resize_size}")

        self.resize_size = int(resize_size)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, self.resize_size, self.resize_size),
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        # Env observation is HWC uint8 RGB; convert to CHW float32 in [0,1].
        chw = torch.from_numpy(observation).permute(2, 0, 1).contiguous()
        chw = resize(
            chw,
            size=[self.resize_size, self.resize_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        chw = chw.to(dtype=torch.float32).div_(255.0)
        return chw.cpu().numpy()


def validate_phase1_observation(obs: np.ndarray, size: int) -> None:
    """Raise with a clear error if wrapped observation contract is violated."""
    if obs.shape != (3, size, size):
        raise ValueError(f"expected shape (3,{size},{size}), got {obs.shape}")
    if obs.dtype != np.float32:
        raise ValueError(f"expected dtype float32, got {obs.dtype}")
    obs_min = float(obs.min())
    obs_max = float(obs.max())
    if obs_min < 0.0 or obs_max > 1.0:
        raise ValueError(f"expected values in [0,1], got min={obs_min}, max={obs_max}")


def make_wrapped_chase_env(config: dict[str, Any]) -> gym.Env:
    """Build a Chase env and apply observation preprocessing from config."""
    env_cfg = config["env"]
    wrapper_cfg = config["wrapper"]

    # Import side effect registers Chase-v0.
    import env  # noqa: F401

    env = gym.make(
        env_cfg["id"],
        n=int(env_cfg["n"]),
        max_episode_steps=env_cfg.get("max_episode_steps"),
        render_mode=env_cfg.get("render_mode"),
    )
    return ChaseObservationWrapper(env, resize_size=int(wrapper_cfg["resize"]))
