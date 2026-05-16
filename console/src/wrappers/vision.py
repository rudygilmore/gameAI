"""Vision preprocessing for the Universal Retro-DQN.

Pipeline (applied in order inside :class:`VisionWrapper`):

1. **Grayscale**: convert ``(H, W, 3)`` uint8 RGB frames to ``(H, W)`` uint8 luma
   using BT.601 weights. This drops two channels but preserves spatial size.
2. **Optional resize**: when ``target_res = (h, w)`` is set in the run TOML,
   resize with OpenCV's bilinear interpolation. When unset the env's native
   raster size is used (per the Decision Log of 2026-05-07).
3. **Frame stack**: maintain a deque of the last ``frame_stack`` processed
   frames and emit them stacked on the leading axis as ``(frame_stack, H, W)``
   uint8. The CNN in :mod:`src.network` consumes channel-first input.

Channel-order convention: outputs are channel-first ``(C, H, W)``. The CNN
infers its flattened linear input size from a dummy forward pass on the same
shape, so changing games or toggling resize never breaks the network.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np

try:
    import cv2

    _HAS_CV2 = True
except ImportError:  # pragma: no cover - exercised only on broken installs
    cv2 = None  # type: ignore[assignment]
    _HAS_CV2 = False


_GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def rgb_to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert an ``(H, W, 3)`` uint8 RGB frame to ``(H, W)`` uint8 luma."""

    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3) RGB frame, got shape {frame.shape}")
    f = frame.astype(np.float32)
    luma = f @ _GRAY_WEIGHTS
    return np.clip(luma, 0, 255).astype(np.uint8)


def resize_frame(frame: np.ndarray, target_res: tuple[int, int]) -> np.ndarray:
    """Resize an ``(H, W)`` uint8 frame to ``target_res = (h, w)``.

    Uses OpenCV's bilinear interpolation when available; falls back to a pure
    NumPy nearest-neighbor resize so unit tests can run on machines without
    OpenCV (CI minimal installs).
    """

    h, w = target_res
    if frame.ndim != 2:
        raise ValueError(f"expected (H, W) gray frame, got shape {frame.shape}")
    if _HAS_CV2:
        return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    src_h, src_w = frame.shape
    ys = (np.linspace(0, src_h - 1, h)).round().astype(np.int64)
    xs = (np.linspace(0, src_w - 1, w)).round().astype(np.int64)
    return frame[np.ix_(ys, xs)].astype(np.uint8, copy=False)


class VisionWrapper(gym.ObservationWrapper):
    """Grayscale -> optional resize -> frame stack, emitted channel-first.

    The wrapper assumes the underlying env produces uint8 RGB frames in
    ``(H, W, 3)``. stable-retro satisfies that contract.
    """

    def __init__(
        self,
        env: gym.Env,
        frame_stack: int = 4,
        target_res: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(env)
        if frame_stack < 1:
            raise ValueError("frame_stack must be >= 1")
        self._frame_stack = int(frame_stack)
        self._target_res = (int(target_res[0]), int(target_res[1])) if target_res else None

        base_shape = env.observation_space.shape
        if base_shape is None or len(base_shape) != 3 or base_shape[-1] != 3:
            raise ValueError(
                f"VisionWrapper expects (H, W, 3) RGB obs, got {base_shape}"
            )
        if self._target_res is not None:
            h, w = self._target_res
        else:
            h, w = int(base_shape[0]), int(base_shape[1])
        self._spatial: tuple[int, int] = (h, w)
        self._frames: deque[np.ndarray] = deque(maxlen=self._frame_stack)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._frame_stack, h, w),
            dtype=np.uint8,
        )

    @property
    def frame_stack(self) -> int:
        return self._frame_stack

    @property
    def spatial(self) -> tuple[int, int]:
        return self._spatial

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = rgb_to_gray(frame)
        if self._target_res is not None:
            gray = resize_frame(gray, self._target_res)
        return gray

    def observation(self, observation: np.ndarray) -> np.ndarray:
        processed = self._preprocess(observation)
        self._frames.append(processed)
        while len(self._frames) < self._frame_stack:
            self._frames.append(processed)
        return np.stack(list(self._frames), axis=0)

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        self._frames.clear()
        obs, info = self.env.reset(**kwargs)
        stacked = self.observation(obs)
        return stacked, info
