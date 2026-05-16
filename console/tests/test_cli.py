"""Smoke tests for ``main.py`` argparse wiring."""

from __future__ import annotations

import pytest

import main as cli


def test_parser_accepts_train_subcommand():
    parser = cli._build_parser()
    ns = parser.parse_args(
        ["train", "--config", "configs/example_run.toml", "--max-steps", "10"]
    )
    assert ns.command == "train"
    assert str(ns.config) == "configs/example_run.toml"
    assert ns.max_steps == 10
    assert ns.seed is None and ns.device is None


def test_parser_accepts_play_subcommand_with_overrides():
    parser = cli._build_parser()
    ns = parser.parse_args(
        [
            "play",
            "--config",
            "configs/example_run.toml",
            "--checkpoint",
            "models/foo.pt",
            "--seed",
            "7",
            "--device",
            "mps",
            "--max-steps",
            "100",
            "--render",
        ]
    )
    assert ns.command == "play"
    assert ns.seed == 7 and ns.device == "mps"
    assert ns.render is True


def test_parser_requires_subcommand():
    parser = cli._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
