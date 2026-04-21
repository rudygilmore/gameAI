from __future__ import annotations

import yaml

from src.wrappers import make_wrapped_chase_env, validate_phase1_observation


def test_phase1_wrapper_contract_from_config() -> None:
    with open("config.yaml", "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    env = make_wrapped_chase_env(config)
    try:
        obs, _info = env.reset(seed=int(config["training"]["seed"]))
        validate_phase1_observation(obs, size=int(config["wrapper"]["resize"]))
    finally:
        env.close()
