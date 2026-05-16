"""CLI entry point for the Universal Retro-DQN.

Usage (run from ``console/`` per the Decision Log entry of 2026-05-07):

    python main.py train --config configs/example_run.toml
    python main.py play  --config configs/example_run.toml --checkpoint models/<stem>.pt

Optional flags allow overriding ``--device`` (cpu/cuda/mps), ``--seed``, and
``--max-steps`` without editing the run TOML; the override is applied on top
of the parsed :class:`~src.config.RunConfig` and never persisted back.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from src.config import RunConfig, load_run_config
from src.play import play as run_play
from src.train import train as run_train


def _override_run_config(
    config: RunConfig,
    *,
    seed: int | None,
    device: str | None,
) -> RunConfig:
    """Return a copy of ``config`` with optional CLI overrides applied."""

    if seed is None and device is None:
        return config
    training = config.training
    if seed is not None:
        training = dataclasses.replace(training, seed=int(seed))
    if device is not None:
        training = dataclasses.replace(training, device=str(device))
    return dataclasses.replace(config, training=training)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description=(
            "Universal Retro-DQN CLI. One TOML run config drives the env, "
            "network, and training hyperparameters; see "
            "EXECPLAN_UNIVERSAL_RETRO_DQN.md for the schema."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train a DQN agent on a stable-retro game.")
    train_p.add_argument("--config", required=True, type=Path, help="Path to the run TOML.")
    train_p.add_argument("--seed", type=int, default=None, help="Override [training].seed.")
    train_p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override [training].device (cpu/cuda/mps).",
    )
    train_p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override [training].total_steps for short smoke runs.",
    )
    train_p.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Print a one-line summary every N optimizer steps (default 1000).",
    )

    play_p = sub.add_parser("play", help="Play with a trained checkpoint (greedy + epsilon).")
    play_p.add_argument("--config", required=True, type=Path, help="Path to the run TOML.")
    play_p.add_argument("--checkpoint", required=True, type=Path, help="Path to a .pt checkpoint.")
    play_p.add_argument("--seed", type=int, default=None, help="Override [training].seed.")
    play_p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override [training].device (cpu/cuda/mps).",
    )
    play_p.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Exploration epsilon for inference rollouts (default 0.05).",
    )
    play_p.add_argument(
        "--max-steps",
        type=int,
        default=10_000,
        help="Maximum environment steps before stopping (default 10000).",
    )
    play_p.add_argument(
        "--render",
        action="store_true",
        help="Render frames via stable-retro's viewer (macOS arm64 patched).",
    )
    play_p.add_argument(
        "--sleep-per-step",
        type=float,
        default=0.0,
        help="Optional sleep (seconds) between steps; useful with --render.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = load_run_config(args.config)
    config = _override_run_config(config, seed=args.seed, device=args.device)

    if args.command == "train":
        path = run_train(config, log_every=args.log_every, max_steps_override=args.max_steps)
        print(str(path))
        return 0

    if args.command == "play":
        total = run_play(
            config,
            args.checkpoint,
            epsilon=args.epsilon,
            max_steps=args.max_steps,
            render=args.render,
            sleep_per_step=args.sleep_per_step,
        )
        print(f"total_reward={total:.4f}")
        return 0

    parser.error(f"unknown command {args.command!r}")
    return 2  # pragma: no cover - parser.error exits


if __name__ == "__main__":
    sys.exit(main())
