from __future__ import annotations

from pathlib import Path

import torch

from src.config_utils import load_config
from src.eval import evaluate
from src.model import MinimalQNet


def test_eval_runs_one_episode_with_random_weights(tmp_path: Path) -> None:
    chase_root = Path(__file__).resolve().parent.parent
    config = load_config(chase_root / "config.yaml")
    ckpt = tmp_path / "policy.pth"
    torch.save(MinimalQNet(4).state_dict(), ckpt)
    config["eval"]["num_episodes"] = 1
    evaluate(config, checkpoint_path=ckpt)


def test_eval_human_mode_with_rgb_array_no_window(tmp_path: Path) -> None:
    """human=True render path without OpenCV (rgb_array)."""
    chase_root = Path(__file__).resolve().parent.parent
    config = load_config(chase_root / "config.yaml")
    ckpt = tmp_path / "policy.pth"
    torch.save(MinimalQNet(4).state_dict(), ckpt)
    config["env"]["render_mode"] = "rgb_array"
    config["eval"]["num_episodes"] = 1
    config["eval"]["render_delay_step"] = 0.0
    config["eval"]["render_delay_episode"] = 0.0
    evaluate(config, checkpoint_path=ckpt, human=True)
