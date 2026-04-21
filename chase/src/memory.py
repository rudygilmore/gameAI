"""Experience replay buffer for DQN."""

from __future__ import annotations

from typing import Tuple

import numpy as np


class ReplayBuffer:
    """Fixed-capacity circular buffer storing transitions as numpy arrays."""

    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], *, seed: int | None = None) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = int(capacity)
        self.obs_shape = tuple(obs_shape)
        if len(self.obs_shape) != 3:
            raise ValueError(f"obs_shape must be (C, H, W), got {self.obs_shape}")

        rng = np.random.default_rng(seed)
        self._rng = rng

        c, h, w = self.obs_shape
        self._obs = np.zeros((capacity, c, h, w), dtype=np.float32)
        self._next_obs = np.zeros((capacity, c, h, w), dtype=np.float32)
        self._actions = np.zeros((capacity,), dtype=np.int64)
        self._rewards = np.zeros((capacity,), dtype=np.float32)
        self._dones = np.zeros((capacity,), dtype=np.bool_)

        self._size = 0
        self._ptr = 0

    def __len__(self) -> int:
        return self._size

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition; **obs** / **next_obs** must match **obs_shape**, float32."""
        if obs.shape != self.obs_shape or next_obs.shape != self.obs_shape:
            raise ValueError(
                f"expected obs shape {self.obs_shape}, got obs={obs.shape}, next_obs={next_obs.shape}"
            )
        i = self._ptr
        self._obs[i] = obs
        self._next_obs[i] = next_obs
        self._actions[i] = int(action)
        self._rewards[i] = float(reward)
        self._dones[i] = bool(done)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample a batch; returns numpy arrays **(obs, actions, rewards, next_obs, dones)**."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if self._size < batch_size:
            raise ValueError(f"buffer has {self._size} transitions, need at least {batch_size}")

        idx = self._rng.integers(0, self._size, size=batch_size, endpoint=False)
        return (
            self._obs[idx].copy(),
            self._actions[idx].copy(),
            self._rewards[idx].copy(),
            self._next_obs[idx].copy(),
            self._dones[idx].copy(),
        )
