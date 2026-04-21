"""DQN building blocks: ε-greedy exploration, Bellman loss vs target net, Adam."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .model import MinimalQNet


def epsilon_exponential(
    global_step: int,
    *,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_steps: float,
) -> float:
    """Exponential decay from **epsilon_start** toward **epsilon_end** as **global_step** grows.

    Uses ``eps = end + (start - end) * exp(-t / tau)`` with ``tau = epsilon_decay_steps``.
    For **t = 0**, returns **epsilon_start**; as **t → ∞**, approaches **epsilon_end** from above.
    """
    if epsilon_decay_steps <= 0:
        raise ValueError(f"epsilon_decay_steps must be positive, got {epsilon_decay_steps}")
    t = float(max(0, global_step))
    tau = float(epsilon_decay_steps)
    span = float(epsilon_start) - float(epsilon_end)
    return float(epsilon_end) + span * math.exp(-t / tau)


def epsilon_greedy_action(
    q_values: torch.Tensor,
    epsilon: float,
    num_actions: int,
    rng: np.random.Generator,
) -> int:
    """Pick an action index given per-action Q-values (1D length **num_actions**)."""
    if q_values.ndim != 1 or q_values.shape[0] != num_actions:
        raise ValueError(f"expected q_values shape ({num_actions},), got {tuple(q_values.shape)}")
    if rng.random() < float(epsilon):
        return int(rng.integers(0, num_actions))
    return int(torch.argmax(q_values).item())


def dqn_bellman_loss(
    policy_net: MinimalQNet,
    target_net: MinimalQNet,
    obs: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_obs: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    """One-step DQN loss: Huber between **Q(s,a)** and **r + γ max_a' Q_target(s',a')** when not terminal."""
    q_all = policy_net(obs)
    q_sa = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_max = target_net(next_obs).max(dim=1).values
        not_done = 1.0 - dones.float()
        target = rewards + float(gamma) * not_done * next_max

    return nn.functional.smooth_l1_loss(q_sa, target)


def build_adam(policy_net: nn.Module, learning_rate: float) -> optim.Adam:
    """Adam optimizer on **policy_net** parameters (**lr** from config, default **1e-4**)."""
    return optim.Adam(policy_net.parameters(), lr=float(learning_rate))


def hard_update_target(policy_net: MinimalQNet, target_net: MinimalQNet) -> None:
    """Copy policy weights into the target network (used every **C** environment steps in training)."""
    target_net.load_state_dict(policy_net.state_dict())


def replay_batch_to_tensors(
    batch: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a **ReplayBuffer.sample** batch to tensors on **device**."""
    obs, actions, rewards, next_obs, dones = batch
    obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32)
    next_obs_t = torch.from_numpy(next_obs).to(device=device, dtype=torch.float32)
    actions_t = torch.from_numpy(actions).to(device=device, dtype=torch.int64)
    rewards_t = torch.from_numpy(rewards).to(device=device, dtype=torch.float32)
    dones_t = torch.from_numpy(dones.astype(np.float32)).to(device=device, dtype=torch.float32)
    return obs_t, actions_t, rewards_t, next_obs_t, dones_t
