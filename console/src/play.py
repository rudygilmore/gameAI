"""Inference entry point for the Universal Retro-DQN.

``play()`` loads a checkpoint into the same architecture described by the run
TOML and steps the environment greedily (with a small exploration epsilon)
for ``max_steps`` interactions. Rendering is optional; the pyglet/Cocoa
workaround captured in ``EXECPLAN_UNIVERSAL_RETRO_DQN.md`` ("Surprises &
Discoveries") is installed at import time of :mod:`src.environment`, so we
just need to pass ``render_mode='human'`` to ``stable_retro.make`` when
``render`` is requested.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .agent import DQNAgent
from .config import RunConfig
from .environment import make_env
from .network import DQN


def play(
    config: RunConfig,
    checkpoint_path: str | Path,
    *,
    epsilon: float = 0.05,
    max_steps: int = 10_000,
    render: bool = False,
    sleep_per_step: float = 0.0,
) -> float:
    """Run a greedy(ish) inference rollout. Returns total reward observed."""

    env = make_env(config, render_mode="human" if render else None)
    try:
        num_actions = int(env.action_space.n)  # type: ignore[attr-defined]
        obs_shape = tuple(env.observation_space.shape)  # type: ignore[arg-type]
        if len(obs_shape) != 3:
            raise RuntimeError(
                f"expected (C, H, W) observation, got shape {obs_shape!r}"
            )

        online = DQN(input_shape=obs_shape, num_actions=num_actions, dueling=config.dueling)
        target = DQN(input_shape=obs_shape, num_actions=num_actions, dueling=config.dueling)
        agent = DQNAgent(
            online=online,
            target=target,
            num_actions=num_actions,
            gamma=config.training.gamma,
            learning_rate=config.training.learning_rate,
            replay_capacity=1,  # play() never trains
            double=config.double,
            device=config.training.device,
            seed=config.training.seed,
        )

        # weights_only=False is required because we serialize the run config
        # alongside the state dicts (a plain Python dict). The file is one
        # we wrote ourselves under console/models/, so the additional pickle
        # surface is acceptable.
        ckpt = torch.load(Path(checkpoint_path), map_location=agent.device, weights_only=False)
        agent.online.load_state_dict(ckpt["online_state_dict"])
        agent.target.load_state_dict(ckpt.get("target_state_dict", ckpt["online_state_dict"]))
        agent.online.eval()

        obs, _ = env.reset(seed=config.training.seed)
        total = 0.0
        for _ in range(max_steps):
            action = agent.select_action(obs, epsilon)
            obs, reward, terminated, truncated, _info = env.step(action)
            total += float(reward)
            # stable-retro auto-renders inside step() when render_mode='human'.
            if sleep_per_step > 0:
                time.sleep(sleep_per_step)
            if terminated or truncated:
                obs, _ = env.reset()
        # Convert to a plain float so the printed return is JSON-friendly.
        return float(np.asarray(total).item())
    finally:
        env.close()
