"""``MultiBinary`` to ``Discrete`` action-space reducer for stable-retro envs.

stable-retro exposes a 12-button ``MultiBinary`` action vector (the joypad).
Standard DQN expects a finite ``Discrete(n)`` action space, so this wrapper
maps a small set of button combinations to discrete indices.

Default combo set (no game-specific tuning required):

* index 0: NOOP (no buttons pressed)
* indices 1..k: each individual button as a one-hot press

Per-game tuning can be added later by passing ``combos=[["LEFT"], ["RIGHT"], ...]``
to :class:`Discretizer`. Until that is wired into the run TOML schema, the
default covers most arcade-style controls without losing single-button actions.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import gymnasium as gym
import numpy as np


def default_combos(buttons: Sequence[str]) -> list[list[str]]:
    """Return the default discrete combo list: NOOP + each individual button.

    The order matches ``buttons`` so a downstream caller can predict indices
    deterministically from the button list reported by stable-retro.
    """

    combos: list[list[str]] = [[]]
    for b in buttons:
        combos.append([b])
    return combos


class Discretizer(gym.ActionWrapper):
    """Wrap an env with ``MultiBinary`` actions and expose ``Discrete(n)``.

    Parameters
    ----------
    env:
        A stable-retro ``RetroEnv`` (or any env exposing
        ``unwrapped.buttons`` and ``MultiBinary`` action space).
    combos:
        Optional sequence of combos; each combo is a list of button names that
        will all be set to 1 when the corresponding discrete index is chosen.
        ``None`` selects :func:`default_combos` for the env's button list.
    """

    def __init__(
        self,
        env: gym.Env,
        combos: Iterable[Iterable[str]] | None = None,
    ) -> None:
        super().__init__(env)
        buttons = self._infer_buttons(env)
        self._buttons: tuple[str, ...] = tuple(buttons)

        combo_list = (
            [list(c) for c in combos] if combos is not None else default_combos(self._buttons)
        )
        if not combo_list:
            raise ValueError("Discretizer requires at least one combo (e.g. NOOP)")

        unknown: set[str] = set()
        for combo in combo_list:
            for name in combo:
                if name not in self._buttons:
                    unknown.add(name)
        if unknown:
            raise ValueError(
                f"Discretizer combos reference unknown buttons {sorted(unknown)}; "
                f"env only exposes {list(self._buttons)}"
            )

        self._combos: tuple[tuple[str, ...], ...] = tuple(tuple(c) for c in combo_list)
        self._table = self._build_action_table()
        self.action_space = gym.spaces.Discrete(len(self._combos))

    @staticmethod
    def _infer_buttons(env: gym.Env) -> Sequence[str]:
        unwrapped = getattr(env, "unwrapped", env)
        buttons = getattr(unwrapped, "buttons", None)
        if buttons is None:
            buttons = getattr(env, "buttons", None)
        if buttons is None:
            raise ValueError(
                "Discretizer needs an env with `.buttons` (e.g. stable-retro RetroEnv)"
            )
        return list(buttons)

    def _build_action_table(self) -> np.ndarray:
        n = len(self._combos)
        table = np.zeros((n, len(self._buttons)), dtype=np.int8)
        index = {name: i for i, name in enumerate(self._buttons)}
        for row, combo in enumerate(self._combos):
            for name in combo:
                table[row, index[name]] = 1
        return table

    @property
    def buttons(self) -> tuple[str, ...]:
        return self._buttons

    @property
    def combos(self) -> tuple[tuple[str, ...], ...]:
        return self._combos

    def action(self, action: int) -> np.ndarray:
        if not 0 <= int(action) < len(self._combos):
            raise IndexError(
                f"discrete action {action} out of range [0, {len(self._combos)})"
            )
        return self._table[int(action)].copy()
