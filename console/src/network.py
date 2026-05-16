"""DQN network with optional dueling head.

The convolutional torso follows the canonical DQN-Nature architecture
(Mnih et al., 2015) but the flattened linear input size is computed from a
dummy forward pass on a tensor matching the post-wrapper observation shape, so
the same module works for arbitrary frame stacks and spatial sizes (Atari 84x84,
NES 224x256, Genesis 224x320, optional ``target_res``, etc.).

When ``dueling = True`` the final layer is split into a value stream and an
advantage stream combined with the dueling aggregation::

    Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))

When ``dueling = False`` the head is a single linear layer over the discrete
action set.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DQN(nn.Module):
    """Convolutional Q-network with optional dueling head."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_actions: int,
        dueling: bool = False,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError(
                f"input_shape must be (C, H, W); got {input_shape!r}"
            )
        self._input_shape = tuple(int(x) for x in input_shape)
        self._num_actions = int(num_actions)
        self._dueling = bool(dueling)

        c, _, _ = self._input_shape
        self.features = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *self._input_shape)
            feat_dim = self.features(dummy).reshape(1, -1).shape[1]
        self._feat_dim = int(feat_dim)

        if self._dueling:
            self.value_head = nn.Sequential(
                nn.Linear(self._feat_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )
            self.advantage_head = nn.Sequential(
                nn.Linear(self._feat_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, self._num_actions),
            )
        else:
            self.q_head = nn.Sequential(
                nn.Linear(self._feat_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, self._num_actions),
            )

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return self._input_shape  # type: ignore[return-value]

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def dueling(self) -> bool:
        return self._dueling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        # Inputs from the env are uint8 in [0, 255]; downstream callers
        # (agent.py) are expected to scale to [0, 1] before forwarding, but we
        # also tolerate already-scaled inputs by leaving values untouched.
        feats = self.features(x).reshape(x.shape[0], -1)
        if self._dueling:
            value = self.value_head(feats)
            advantage = self.advantage_head(feats)
            advantage_centered = advantage - advantage.mean(dim=1, keepdim=True)
            return value + advantage_centered
        return self.q_head(feats)
