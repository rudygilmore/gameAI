"""Environment wrappers for the Universal Retro-DQN.

This subpackage contains exactly three modules at v1 (per the Decision Log
entry of 2026-05-15 in ``EXECPLAN_UNIVERSAL_RETRO_DQN.md``):

* :mod:`src.wrappers.discretizer` -- ``MultiBinary`` to ``Discrete`` reducer.
* :mod:`src.wrappers.vision` -- grayscale, optional resize, frame stack.
* :mod:`src.wrappers.reward_wrapper` -- :class:`RewardWrapper` base class for
  per-game reward shaping modules under ``console/configs/rewards/``.
"""

from .discretizer import Discretizer
from .reward_wrapper import RewardWrapper
from .vision import VisionWrapper

__all__ = ["Discretizer", "RewardWrapper", "VisionWrapper"]
