"""Training loop and checkpoint helpers for the Universal Retro-DQN.

Checkpoints are written under ``console/models/`` with the stem rule from the
Decision Log entry of 2026-05-07::

    {game_slug}_d{0|1}_u{0|1}_v{YYYYMMDDTHHMMSSZ}.pt

The slug function is :func:`game_slug`; the timestamp is UTC truncated to
seconds; the boolean flags ``d`` (double Q-learning) and ``u`` (dueling head;
mnemonic 'u' for the dueling 'u' flag) reflect the run's TOML.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .agent import DQNAgent, Transition, linear_epsilon
from .config import RunConfig
from .environment import make_env
from .network import DQN


_SLUG_FORBIDDEN = re.compile(r"[^A-Za-z0-9._-]+")


def game_slug(game_id: str) -> str:
    """Replace path separators and other illegal filename characters with ``_``."""

    if not game_id:
        raise ValueError("game id must be a non-empty string")
    return _SLUG_FORBIDDEN.sub("_", game_id).strip("_") or "game"


def make_checkpoint_stem(
    game_id: str, double: bool, dueling: bool, when: datetime
) -> str:
    """Return the checkpoint filename stem (no extension).

    ``when`` is converted to UTC and truncated to whole seconds before
    formatting as ``YYYYMMDDTHHMMSSZ``.
    """

    if when.tzinfo is None:
        utc = when.replace(tzinfo=timezone.utc)
    else:
        utc = when.astimezone(timezone.utc)
    utc = utc.replace(microsecond=0)
    timestamp = utc.strftime("%Y%m%dT%H%M%SZ")
    return (
        f"{game_slug(game_id)}_d{int(bool(double))}_u{int(bool(dueling))}_v{timestamp}"
    )


def _set_global_seeds(seed: int) -> None:
    import random as _py_random

    _py_random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - hardware dependent
        torch.cuda.manual_seed_all(seed)


def _build_agent(config: RunConfig, obs_shape: tuple[int, int, int], num_actions: int) -> DQNAgent:
    online = DQN(input_shape=obs_shape, num_actions=num_actions, dueling=config.dueling)
    target = DQN(input_shape=obs_shape, num_actions=num_actions, dueling=config.dueling)
    return DQNAgent(
        online=online,
        target=target,
        num_actions=num_actions,
        gamma=config.training.gamma,
        learning_rate=config.training.learning_rate,
        replay_capacity=config.training.replay_capacity,
        double=config.double,
        device=config.training.device,
        seed=config.training.seed,
    )


def _models_dir(config: RunConfig) -> Path:
    """Locate ``models/`` next to ``configs/`` when possible.

    Mirrors the behavior of :func:`src.environment._rewards_dir`: if the run
    TOML lives at ``console/configs/<file>.toml`` we write under
    ``console/models/``; otherwise we fall back to ``models/`` under cwd.
    """

    if config.source_path is not None:
        anchor = config.source_path.parent
        candidate = anchor.parent / "models"
        if candidate.exists() or candidate.parent.is_dir():
            return candidate
    return Path.cwd() / "models"


def train(
    config: RunConfig,
    *,
    log_every: int = 1000,
    max_steps_override: int | None = None,
) -> Path:
    """Run a full training session and return the path to the saved checkpoint."""

    _set_global_seeds(config.training.seed)
    env = make_env(config)
    try:
        num_actions = int(env.action_space.n)  # type: ignore[attr-defined]
        obs_shape = tuple(env.observation_space.shape)  # type: ignore[arg-type]
        if len(obs_shape) != 3:
            raise RuntimeError(
                f"expected (C, H, W) observation, got shape {obs_shape!r}"
            )

        agent = _build_agent(config, obs_shape, num_actions)
        total_steps = max_steps_override or config.training.total_steps

        obs, _ = env.reset(seed=config.training.seed)
        episode_return = 0.0
        episode_index = 0
        episode_step = 0
        start_wall = time.time()

        for step in range(1, total_steps + 1):
            epsilon = linear_epsilon(
                step,
                start=config.training.epsilon_start,
                end=config.training.epsilon_end,
                decay_steps=config.training.epsilon_decay_steps,
            )
            action = agent.select_action(obs, epsilon)
            next_obs, reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)
            agent.remember(
                Transition(
                    obs=np.asarray(obs, dtype=np.uint8).copy(),
                    action=int(action),
                    reward=float(reward),
                    next_obs=np.asarray(next_obs, dtype=np.uint8).copy(),
                    done=done,
                )
            )
            episode_return += float(reward)
            episode_step += 1

            loss = None
            if agent.can_train(config.training.batch_size):
                loss = agent.train_step(config.training.batch_size)

            if step % config.training.target_update_period == 0:
                agent.sync_target()

            if step % log_every == 0:
                elapsed = time.time() - start_wall
                fps = step / elapsed if elapsed > 0 else float("inf")
                loss_str = "n/a" if loss is None else f"{loss:.4f}"
                print(
                    f"[step {step}/{total_steps}] eps={epsilon:.3f} loss={loss_str} "
                    f"ep={episode_index} fps={fps:.1f}",
                    flush=True,
                )

            if done:
                print(
                    f"[episode {episode_index} done] return={episode_return:.2f} "
                    f"length={episode_step} step={step}",
                    flush=True,
                )
                obs, _ = env.reset()
                episode_return = 0.0
                episode_step = 0
                episode_index += 1
            else:
                obs = next_obs

        checkpoint_path = save_checkpoint(agent, config, datetime.now(timezone.utc))
        print(f"[done] checkpoint saved to {checkpoint_path}", flush=True)
        return checkpoint_path
    finally:
        env.close()


def save_checkpoint(agent: DQNAgent, config: RunConfig, when: datetime) -> Path:
    stem = make_checkpoint_stem(config.game, config.double, config.dueling, when)
    out_dir = _models_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.pt"
    payload: dict[str, Any] = {
        "online_state_dict": agent.online.state_dict(),
        "target_state_dict": agent.target.state_dict(),
        "config": {
            "game": config.game,
            "state": config.state,
            "double": config.double,
            "dueling": config.dueling,
            "frame_stack": config.frame_stack,
            "target_res": config.target_res,
        },
    }
    torch.save(payload, path)
    return path
