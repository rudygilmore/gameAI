"""Shape and architecture tests for ``src.network.DQN``."""

from __future__ import annotations

import pytest
import torch

from src.network import DQN


@pytest.mark.parametrize(
    "shape",
    [
        (4, 84, 84),
        (4, 224, 320),  # Genesis native
        (2, 96, 128),
        (1, 64, 64),
    ],
)
@pytest.mark.parametrize("dueling", [False, True])
def test_forward_shape(shape, dueling):
    net = DQN(input_shape=shape, num_actions=13, dueling=dueling)
    x = torch.zeros(7, *shape)
    y = net(x)
    assert y.shape == (7, 13)


def test_forward_handles_uint8_inputs():
    net = DQN(input_shape=(4, 84, 84), num_actions=6, dueling=False)
    x = torch.zeros(3, 4, 84, 84, dtype=torch.uint8)
    y = net(x)
    assert y.shape == (3, 6) and y.dtype == torch.float32


def test_dueling_aggregation_centers_advantage():
    """When dueling=True the advantage stream contributes mean-zero across actions."""

    net = DQN(input_shape=(2, 84, 84), num_actions=4, dueling=True)
    x = torch.randn(5, 2, 84, 84)
    q = net(x)
    # Manually compute the value stream and check Q - V matches centered
    # advantage (mean across actions == 0 by the aggregation formula).
    feats = net.features(x.float()).reshape(x.shape[0], -1)
    v = net.value_head(feats)  # (5, 1)
    centered = q - v
    assert torch.allclose(centered.mean(dim=1), torch.zeros(5), atol=1e-5)


def test_invalid_input_shape_raises():
    with pytest.raises(ValueError):
        DQN(input_shape=(84, 84), num_actions=4)  # type: ignore[arg-type]
