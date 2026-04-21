from __future__ import annotations

import math

import numpy as np
import torch

from src.dqn import (
    build_adam,
    dqn_bellman_loss,
    epsilon_exponential,
    epsilon_greedy_action,
    hard_update_target,
    replay_batch_to_tensors,
)
from src.model import make_policy_and_target


def test_epsilon_exponential_endpoints() -> None:
    start, end, decay = 1.0, 0.05, 10000.0
    assert abs(epsilon_exponential(0, epsilon_start=start, epsilon_end=end, epsilon_decay_steps=decay) - start) < 1e-6
    late = epsilon_exponential(1_000_000, epsilon_start=start, epsilon_end=end, epsilon_decay_steps=decay)
    assert late < start and late >= end and abs(late - end) < 0.01


def test_epsilon_exponential_decay_curve() -> None:
    """Later steps should yield smaller epsilon (monotone decay toward end)."""
    start, end, decay = 1.0, 0.05, 1000.0
    e0 = epsilon_exponential(0, epsilon_start=start, epsilon_end=end, epsilon_decay_steps=decay)
    e1 = epsilon_exponential(500, epsilon_start=start, epsilon_end=end, epsilon_decay_steps=decay)
    e2 = epsilon_exponential(5000, epsilon_start=start, epsilon_end=end, epsilon_decay_steps=decay)
    assert e0 > e1 > e2


def test_epsilon_greedy_deterministic_when_zero() -> None:
    rng = np.random.default_rng(0)
    q = torch.tensor([0.0, 1.0, 0.0, 0.0])
    for _ in range(20):
        assert epsilon_greedy_action(q, epsilon=0.0, num_actions=4, rng=rng) == 1


def test_dqn_loss_backward() -> None:
    device = torch.device("cpu")
    policy, target = make_policy_and_target(num_actions=4, device=device)
    b = 8
    obs = torch.randn(b, 3, 64, 64, device=device)
    next_obs = torch.randn(b, 3, 64, 64, device=device)
    actions = torch.zeros(b, dtype=torch.int64, device=device)
    rewards = torch.randn(b, device=device)
    dones = torch.zeros(b, device=device, dtype=torch.float32)

    loss = dqn_bellman_loss(
        policy, target, obs, actions, rewards, next_obs, dones, gamma=0.99
    )
    assert loss.ndim == 0 and torch.isfinite(loss)
    opt = build_adam(policy, learning_rate=1e-4)
    opt.zero_grad()
    loss.backward()
    opt.step()


def test_hard_update_makes_equal_outputs() -> None:
    device = torch.device("cpu")
    policy, target = make_policy_and_target(num_actions=4, device=device)
    policy.head.bias.data.add_(1.0)
    x = torch.randn(2, 3, 64, 64, device=device)
    assert not torch.allclose(policy(x), target(x))
    hard_update_target(policy, target)
    assert torch.allclose(policy(x), target(x))


def test_replay_batch_to_tensors_roundtrip() -> None:
    obs = np.zeros((4, 3, 64, 64), dtype=np.float32)
    next_obs = np.ones((4, 3, 64, 64), dtype=np.float32)
    actions = np.array([0, 1, 2, 3], dtype=np.int64)
    rewards = np.array([1.0, 0.0, -1.0, 0.5], dtype=np.float32)
    dones = np.array([False, True, False, False], dtype=np.bool_)
    batch = (obs, actions, rewards, next_obs, dones)
    dev = torch.device("cpu")
    ot, at, rt, nt, dt = replay_batch_to_tensors(batch, dev)
    assert ot.shape == (4, 3, 64, 64) and nt.shape == (4, 3, 64, 64)
    assert at.dtype == torch.int64 and dt.dtype == torch.float32
    assert float(dt[1].item()) == 1.0  # done
