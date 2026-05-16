"""Base class for per-game reward shaping wrappers.

Each module under ``console/configs/rewards/`` (when ``reward_shaping = true``
in the run TOML) must export exactly one public class named ``Wrapper`` that
subclasses :class:`RewardWrapper`. The environment factory in
:mod:`src.environment` instantiates it as ``mod.Wrapper(env)`` (see the
"Reward extension interface" section of
``console/EXECPLAN_UNIVERSAL_RETRO_DQN.md``).

The implementation here is a thin alias over :class:`gymnasium.RewardWrapper`
so subclasses only need to override :meth:`reward`. We expose it under our own
namespace to keep the per-game module imports concise:

    from src.wrappers.reward_wrapper import RewardWrapper

    class Wrapper(RewardWrapper):
        def reward(self, reward):
            return reward * 0.1
"""

from __future__ import annotations

import gymnasium as gym


class RewardWrapper(gym.RewardWrapper):
    """Per-game reward shaping base. Subclass and override :meth:`reward`."""

    # The whole API surface we promise is gymnasium.RewardWrapper's. Subclasses
    # may also override `step`/`reset` if they need to read `info` from the
    # underlying env, but the minimal contract is just `reward(self, reward)`.
    pass
