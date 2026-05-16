# console/configs/rewards/example_scaling.py
# Minimal example reward module. Real per-game shaping should subclass
# `RewardWrapper` and (optionally) override `step` to read the env's `info`
# dict for game-specific signals. See EXECPLAN_UNIVERSAL_RETRO_DQN.md
# (Decision Log 2026-05-15) for the convention.
from src.wrappers.reward_wrapper import RewardWrapper


class Wrapper(RewardWrapper):
    SCALE = 0.1

    def reward(self, reward: float) -> float:
        return float(reward) * self.SCALE
