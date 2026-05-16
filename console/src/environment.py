"""Environment factory: ``retro.make`` + wrapper composition.

Composition order (fixed for v1, per Milestone B in
``console/EXECPLAN_UNIVERSAL_RETRO_DQN.md``):

1. ``stable_retro.make(game=..., state=...)``
2. :class:`src.wrappers.discretizer.Discretizer` -> ``Discrete(n)`` action space
3. :class:`src.wrappers.vision.VisionWrapper` -> grayscale, optional resize,
   frame stack, channel-first ``(C, H, W)`` uint8 observations
4. Optional reward shaping via the ``Wrapper`` class in
   ``console/configs/rewards/<reward_module>.py``

The spatial shape used by the CNN comes from the post-wrapper observation
space, so the CNN never needs to know the env's native raster dimensions.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium as gym

from .config import RunConfig
from .wrappers.discretizer import Discretizer
from .wrappers.reward_wrapper import RewardWrapper
from .wrappers.vision import VisionWrapper

if TYPE_CHECKING:  # pragma: no cover
    pass


def _install_pyglet_cocoa_workaround() -> None:
    """Patch pyglet's ``CocoaAlternateEventLoop.exit`` on macOS arm64.

    See "Surprises & Discoveries" in ``EXECPLAN_UNIVERSAL_RETRO_DQN.md``.
    The patch is installed unconditionally on darwin so even users who do not
    request rendering survive ``env.close()`` after stable-retro's
    ``render_mode='human'`` default has spawned a viewer. The patch is
    idempotent and a no-op when ``pyglet.app.cocoa`` cannot be imported.
    """

    if sys.platform != "darwin":
        return
    try:
        from pyglet.app import cocoa as _pyglet_cocoa
    except Exception:  # pragma: no cover - non-cocoa platform
        return
    cls = getattr(_pyglet_cocoa, "CocoaAlternateEventLoop", None)
    if cls is None or getattr(cls, "_console_patched", False):
        return
    _orig_exit = cls.exit

    def _patched_exit(self):  # type: ignore[no-untyped-def]
        if getattr(self, "platform_event_loop", None) is None:
            import pyglet.app as _app

            self.platform_event_loop = _app.platform_event_loop
        return _orig_exit(self)

    cls.exit = _patched_exit  # type: ignore[assignment]
    cls._console_patched = True  # type: ignore[attr-defined]


_install_pyglet_cocoa_workaround()


def _rewards_dir(config: RunConfig) -> Path:
    """Locate ``configs/rewards/`` relative to the run TOML when possible.

    Falls back to ``configs/rewards/`` under the current working directory.
    The ExecPlan's Decision Log (2026-05-07) sets ``console/`` as the working
    directory, so the fallback also resolves to the right tree when invoked
    via ``python main.py …`` from ``console/``.
    """

    if config.source_path is not None:
        anchor = config.source_path.parent
        # Run TOMLs typically live at console/configs/<file>.toml; rewards is
        # the sibling subdirectory next to them.
        candidate = anchor / "rewards"
        if candidate.is_dir():
            return candidate
        # Fall back to anchor/../configs/rewards if the run TOML lives elsewhere.
        candidate = anchor.parent / "configs" / "rewards"
        if candidate.is_dir():
            return candidate
    return Path.cwd() / "configs" / "rewards"


def _load_reward_wrapper_class(config: RunConfig) -> type[RewardWrapper]:
    assert config.reward_module is not None  # checked by config loader
    rewards_dir = _rewards_dir(config)
    module_path = rewards_dir / f"{config.reward_module}.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            f"reward_module '{config.reward_module}' not found at {module_path}"
        )
    spec_name = f"_console_reward_{config.reward_module}"
    spec = importlib.util.spec_from_file_location(spec_name, module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib invariant
        raise ImportError(f"could not load reward module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, "Wrapper", None)
    if cls is None:
        raise AttributeError(
            f"reward module {module_path} must define a public class named 'Wrapper' "
            "(see Decision Log entry of 2026-05-15)"
        )
    if not (isinstance(cls, type) and issubclass(cls, RewardWrapper)):
        raise TypeError(
            f"'Wrapper' in {module_path} must subclass src.wrappers.reward_wrapper.RewardWrapper"
        )
    return cls


def make_env(config: RunConfig, *, render_mode: str | None = None) -> gym.Env:
    """Build the wrapped stable-retro env described by ``config``.

    ``render_mode`` is passed through to ``stable_retro.make`` and defaults to
    ``None`` so neither :meth:`~gym.Env.reset` nor :meth:`~gym.Env.step`
    auto-spawns the pyglet viewer (stable-retro 1.0.0's default is ``"human"``,
    which causes a window to open during training and crashes ``env.close()``
    on macOS arm64 without the workaround installed at import time).
    """

    # Imported lazily so unit tests on machines without a stable-retro install
    # (or without imported ROMs) can still import this module.
    import stable_retro

    make_kwargs: dict = {"game": config.game, "render_mode": render_mode}
    if config.state is not None:
        make_kwargs["state"] = config.state
    env: gym.Env = stable_retro.make(**make_kwargs)

    env = Discretizer(env)
    env = VisionWrapper(env, frame_stack=config.frame_stack, target_res=config.target_res)

    if config.reward_shaping:
        wrapper_cls = _load_reward_wrapper_class(config)
        env = wrapper_cls(env)

    return env
