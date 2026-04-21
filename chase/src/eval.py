"""Evaluate a trained Chase DQN checkpoint (loads **config.yaml**)."""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config_utils import load_config, resolve_training_device, validate_config
from .dqn import epsilon_greedy_action
from .model import MinimalQNet
from .wrappers import make_wrapped_chase_env


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Chase DQN from a checkpoint.")
    p.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML (default: config.yaml in cwd)",
    )
    p.add_argument(
        "--human",
        action="store_true",
        help="Open a window and show the agent (sets env render_mode to human; uses eval.render_delay_*)",
    )
    return p.parse_args(argv)


def evaluate(
    config: dict[str, Any],
    *,
    checkpoint_path: Path | None = None,
    human: bool = False,
) -> None:
    """Run **eval.num_episodes** with **eval.epsilon**; print per-episode return and success."""
    validate_config(config)
    train_cfg = config["training"]
    eval_cfg = config["eval"]
    step_delay = float(eval_cfg.get("render_delay_step", 0.05))
    episode_delay = float(eval_cfg.get("render_delay_episode", 0.5))

    seed = int(train_cfg["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 42)

    device = resolve_training_device(str(train_cfg["device"]))
    env = make_wrapped_chase_env(config)
    num_actions = int(env.action_space.n)

    policy = MinimalQNet(num_actions=num_actions).to(device)
    ckpt = Path(checkpoint_path) if checkpoint_path is not None else Path(str(eval_cfg["checkpoint"]))
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt.resolve()}")
    policy.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    policy.eval()

    num_episodes = int(eval_cfg["num_episodes"])
    eval_epsilon = float(eval_cfg["epsilon"])

    successes = 0
    returns: list[float] = []

    try:
        for episode in range(num_episodes):
            if human and episode > 0:
                time.sleep(episode_delay)

            obs, _info = env.reset(seed=seed + 1000 + episode)
            ep_return = 0.0
            terminated = False
            truncated = False

            if human:
                env.render()
                time.sleep(step_delay)

            while not (terminated or truncated):
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    q_flat = policy(obs_t).squeeze(0)
                action = epsilon_greedy_action(q_flat.cpu(), eval_epsilon, num_actions, rng)

                next_obs, reward, terminated, truncated, _info = env.step(action)
                ep_return += float(reward)
                obs = next_obs

                if human:
                    env.render()
                    time.sleep(step_delay)

            returns.append(ep_return)
            success = bool(terminated)
            if success:
                successes += 1
            print(
                f"eval episode={episode + 1}/{num_episodes} return={ep_return:.4f} "
                f"terminated(success)={terminated}",
                flush=True,
            )
    finally:
        env.close()

    rate = successes / max(1, num_episodes)
    mean_ret = float(np.mean(returns)) if returns else 0.0
    print(
        f"summary: success_rate={rate:.2%} ({successes}/{num_episodes}) mean_return={mean_ret:.4f}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Config not found: {config_path.resolve()}", file=sys.stderr)
        sys.exit(1)
    config = load_config(config_path)
    if args.human:
        config = copy.deepcopy(config)
        config["env"]["render_mode"] = "human"
    try:
        evaluate(config, human=args.human)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
