"""Minimal CNN Q-network for discrete Chase actions (4 directions)."""

from __future__ import annotations

import copy
from typing import Tuple

import torch
import torch.nn as nn


class MinimalQNet(nn.Module):
    """Small convolutional Q-network: maps CHW observations to four Q-values.

    Architecture: three stride-2 conv blocks with same padding (targets **64×64** inputs
    from the default wrapper; other square sizes that remain positive through the stack
    also work), global average pool, linear head to **num_actions** (default **4**).
    """

    def __init__(self, num_actions: int = 4) -> None:
        super().__init__()
        self.num_actions = int(num_actions)
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(64, self.num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Q-values of shape **(batch, num_actions)** for input **(B, 3, H, W)**."""
        z = self.features(x)
        z = self.pool(z)
        z = z.flatten(1)
        return self.head(z)


def make_policy_and_target(
    *,
    num_actions: int = 4,
    device: torch.device | str | None = None,
) -> Tuple[MinimalQNet, MinimalQNet]:
    """Instantiate policy and target networks with identical architecture and initial weights."""
    policy = MinimalQNet(num_actions=num_actions)
    target = copy.deepcopy(policy)
    if device is not None:
        dev = device if isinstance(device, torch.device) else torch.device(device)
        policy = policy.to(dev)
        target = target.to(dev)
    return policy, target
