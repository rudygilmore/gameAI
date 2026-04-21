from __future__ import annotations

import torch

from src.model import MinimalQNet, make_policy_and_target


def test_minimal_qnet_output_shape() -> None:
    net = MinimalQNet(num_actions=4)
    b, h, w = 8, 64, 64
    x = torch.randn(b, 3, h, w)
    q = net(x)
    assert q.shape == (b, 4)


def test_make_policy_and_target_same_architecture() -> None:
    policy, target = make_policy_and_target(num_actions=4, device="cpu")
    x = torch.randn(2, 3, 64, 64)
    assert torch.allclose(policy(x), target(x))
