from __future__ import annotations

import numpy as np

from src.memory import ReplayBuffer


def test_replay_sample_shapes_and_dtypes() -> None:
    obs_shape = (3, 64, 64)
    buf = ReplayBuffer(capacity=100, obs_shape=obs_shape, seed=0)
    o = np.random.rand(*obs_shape).astype(np.float32)
    o2 = np.random.rand(*obs_shape).astype(np.float32)
    for _ in range(50):
        buf.push(o, 0, 1.0, o2, False)

    obs, actions, rewards, next_obs, dones = buf.sample(32)
    assert obs.shape == (32, 3, 64, 64) and obs.dtype == np.float32
    assert next_obs.shape == (32, 3, 64, 64)
    assert actions.shape == (32,) and actions.dtype == np.int64
    assert rewards.shape == (32,) and rewards.dtype == np.float32
    assert dones.shape == (32,) and dones.dtype == np.bool_
