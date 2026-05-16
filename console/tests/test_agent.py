"""Tensor-path tests for the DQN agent (no env or ROMs required)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.agent import DQNAgent, ReplayBuffer, Transition, linear_epsilon
from src.network import DQN


# The DQN-Nature torso (kernels 8/4/3) needs at least ~64 px on each spatial
# axis. We use the canonical 84x84 to keep tests realistic.
_TEST_OBS_SHAPE = (4, 84, 84)


def _build_agent(*, double: bool, dueling: bool = False, capacity: int = 64) -> DQNAgent:
    online = DQN(input_shape=_TEST_OBS_SHAPE, num_actions=5, dueling=dueling)
    target = DQN(input_shape=_TEST_OBS_SHAPE, num_actions=5, dueling=dueling)
    return DQNAgent(
        online=online,
        target=target,
        num_actions=5,
        gamma=0.99,
        learning_rate=1e-3,
        replay_capacity=capacity,
        double=double,
        device="cpu",
        seed=0,
    )


def _fake_transition(rng: np.random.Generator) -> Transition:
    obs = (rng.integers(0, 256, size=_TEST_OBS_SHAPE)).astype(np.uint8)
    next_obs = (rng.integers(0, 256, size=_TEST_OBS_SHAPE)).astype(np.uint8)
    return Transition(
        obs=obs,
        action=int(rng.integers(0, 5)),
        reward=float(rng.normal()),
        next_obs=next_obs,
        done=bool(rng.integers(0, 2)),
    )


def test_replay_buffer_overwrites_oldest():
    buf = ReplayBuffer(capacity=3)
    rng = np.random.default_rng(0)
    for _ in range(5):
        buf.push(_fake_transition(rng))
    assert len(buf) == 3
    sampled = buf.sample(2)
    assert len(sampled) == 2


def test_replay_sample_too_large_raises():
    buf = ReplayBuffer(capacity=10)
    rng = np.random.default_rng(0)
    buf.push(_fake_transition(rng))
    with pytest.raises(ValueError):
        buf.sample(2)


def test_linear_epsilon_endpoints():
    assert linear_epsilon(0, start=1.0, end=0.05, decay_steps=100) == pytest.approx(1.0)
    assert linear_epsilon(50, start=1.0, end=0.05, decay_steps=100) == pytest.approx(0.525)
    assert linear_epsilon(100, start=1.0, end=0.05, decay_steps=100) == pytest.approx(0.05)
    assert linear_epsilon(1000, start=1.0, end=0.05, decay_steps=100) == pytest.approx(0.05)
    # decay_steps == 0 collapses immediately to `end`.
    assert linear_epsilon(0, start=1.0, end=0.05, decay_steps=0) == pytest.approx(0.05)


@pytest.mark.parametrize("double", [False, True])
@pytest.mark.parametrize("dueling", [False, True])
def test_train_step_runs_and_updates_parameters(double, dueling):
    torch.manual_seed(0)
    agent = _build_agent(double=double, dueling=dueling)
    rng = np.random.default_rng(0)
    for _ in range(40):
        agent.remember(_fake_transition(rng))
    before = [p.detach().clone() for p in agent.online.parameters()]
    loss = agent.train_step(batch_size=8)
    assert isinstance(loss, float) and loss == loss  # not NaN
    after = list(agent.online.parameters())
    deltas = [
        (b - a).abs().sum().item()
        for b, a in zip(before, after)
        if a.requires_grad
    ]
    assert any(d > 0 for d in deltas), "Optimizer step did not change any params"


def test_select_action_in_range():
    agent = _build_agent(double=False)
    obs = np.zeros(_TEST_OBS_SHAPE, dtype=np.uint8)
    action = agent.select_action(obs, epsilon=0.0)
    assert 0 <= action < 5


def test_sync_target_copies_weights():
    agent = _build_agent(double=False)
    # Perturb online weights then sync; verify target matches.
    with torch.no_grad():
        for p in agent.online.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    agent.sync_target()
    for p_o, p_t in zip(agent.online.parameters(), agent.target.parameters()):
        assert torch.allclose(p_o, p_t)
