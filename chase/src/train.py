"""DQN training loop for Chase (loads **config.yaml**)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config_utils import load_config, resolve_training_device, validate_config
from .dqn import (
    build_adam,
    dqn_bellman_loss,
    epsilon_exponential,
    epsilon_greedy_action,
    hard_update_target,
    replay_batch_to_tensors,
)
from .memory import ReplayBuffer
from .model import make_policy_and_target
from .wrappers import make_wrapped_chase_env


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Chase DQN from a YAML config.")
    p.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML (default: config.yaml in cwd)",
    )
    return p.parse_args(argv)


def train(config: dict[str, Any], *, checkpoint_path: Path | None = None) -> None:
    """Run training; saves policy weights to **eval.checkpoint** (or **checkpoint_path**)."""
    validate_config(config)
    train_cfg = config["training"]
    eval_cfg = config["eval"]
    log_cfg = config["logging"]
    wrapper_cfg = config["wrapper"]

    seed = int(train_cfg["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    device = resolve_training_device(str(train_cfg["device"]))
    resize = int(wrapper_cfg["resize"])
    obs_shape = (3, resize, resize)

    env = make_wrapped_chase_env(config)
    num_actions = int(env.action_space.n)

    policy, target = make_policy_and_target(num_actions=num_actions, device=device)
    optimizer = build_adam(policy, float(train_cfg["learning_rate"]))

    buffer = ReplayBuffer(
        capacity=int(train_cfg["replay_capacity"]),
        obs_shape=obs_shape,
        seed=seed + 1,
    )

    gamma = float(train_cfg["gamma"])
    batch_size = int(train_cfg["batch_size"])
    min_replay = int(train_cfg["min_replay_size"])
    target_every = int(train_cfg["target_update_every"])
    num_episodes = int(train_cfg["num_episodes"])
    eps_start = float(train_cfg["epsilon_start"])
    eps_end = float(train_cfg["epsilon_end"])
    eps_decay_steps = float(train_cfg["epsilon_decay_steps"])
    log_every = int(log_cfg["log_every_episodes"])

    ckpt = Path(checkpoint_path) if checkpoint_path is not None else Path(str(eval_cfg["checkpoint"]))
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    global_step = 0
    ep_returns: list[float] = []
    recent_losses: list[float] = []

    try:
        for episode in range(num_episodes):
            obs, _info = env.reset(seed=seed + episode)
            ep_return = 0.0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                epsilon = epsilon_exponential(
                    global_step,
                    epsilon_start=eps_start,
                    epsilon_end=eps_end,
                    epsilon_decay_steps=eps_decay_steps,
                )

                obs_t = torch.from_numpy(obs).unsqueeze(0).to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    q_flat = policy(obs_t).squeeze(0)
                action = epsilon_greedy_action(q_flat, epsilon, num_actions, rng)

                next_obs, reward, terminated, truncated, _info = env.step(action)
                done = terminated or truncated
                ep_return += float(reward)

                buffer.push(obs, action, float(reward), next_obs, done)
                obs = next_obs
                global_step += 1

                if len(buffer) >= min_replay:
                    batch = buffer.sample(batch_size)
                    obs_b, act_b, rew_b, next_b, done_b = replay_batch_to_tensors(batch, device)
                    optimizer.zero_grad()
                    loss = dqn_bellman_loss(
                        policy,
                        target,
                        obs_b,
                        act_b,
                        rew_b,
                        next_b,
                        done_b,
                        gamma=gamma,
                    )
                    loss.backward()
                    optimizer.step()
                    recent_losses.append(float(loss.detach().cpu().item()))

                if global_step > 0 and global_step % target_every == 0:
                    hard_update_target(policy, target)

            ep_returns.append(ep_return)

            if (episode + 1) % log_every == 0 or episode == 0:
                mean_r = float(np.mean(ep_returns[-log_every :])) if ep_returns else 0.0
                mean_loss = float(np.mean(recent_losses[-1000:])) if recent_losses else float("nan")
                eps_now = epsilon_exponential(
                    global_step,
                    epsilon_start=eps_start,
                    epsilon_end=eps_end,
                    epsilon_decay_steps=eps_decay_steps,
                )
                print(
                    f"episode={episode + 1}/{num_episodes} "
                    f"return_mean_last_{min(log_every, len(ep_returns))}={mean_r:.4f} "
                    f"loss_mean_recent={mean_loss:.6f} "
                    f"epsilon={eps_now:.4f} "
                    f"step={global_step}",
                    flush=True,
                )
    finally:
        env.close()

    torch.save(policy.state_dict(), ckpt)
    print(f"Saved policy state dict to {ckpt}", flush=True)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Config not found: {config_path.resolve()}", file=sys.stderr)
        sys.exit(1)
    config = load_config(config_path)
    train(config)


if __name__ == "__main__":
    main()
