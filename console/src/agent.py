"""DQN agent: replay buffer, epsilon-greedy policy, vanilla/double TD targets.

The agent owns two networks: an online :class:`~src.network.DQN` that is
updated each optimization step and a target network that is periodically synced
from the online weights (every ``target_update_period`` steps in
:mod:`src.train`).

* When ``double = False`` the bootstrap target uses the target network for both
  argmax selection and value evaluation::

      y = r + gamma * (1 - done) * max_a Q_target(s', a)

* When ``double = True`` the online network selects the action and the target
  network evaluates it (Double DQN, van Hasselt et al., 2016)::

      a*  = argmax_a Q_online(s', a)
      y   = r + gamma * (1 - done) * Q_target(s', a*)

The replay buffer stores raw uint8 frames to keep memory low; the float32 cast
and ``/255`` scaling happen at sample time so the conv torso always sees
inputs in roughly ``[0, 1]``.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .network import DQN


@dataclass(frozen=True)
class Transition:
    """One environment transition stored in the replay buffer."""

    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool


class ReplayBuffer:
    """Bounded FIFO replay buffer for uint8 stacked frames."""

    def __init__(self, capacity: int, rng: random.Random | None = None) -> None:
        if capacity < 1:
            raise ValueError("replay capacity must be >= 1")
        self._buf: deque[Transition] = deque(maxlen=capacity)
        self._rng = rng or random.Random()

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def capacity(self) -> int:
        return self._buf.maxlen  # type: ignore[return-value]

    def push(self, t: Transition) -> None:
        self._buf.append(t)

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size > len(self._buf):
            raise ValueError(
                f"requested {batch_size} samples but buffer holds {len(self._buf)}"
            )
        return self._rng.sample(list(self._buf), batch_size)


def linear_epsilon(step: int, *, start: float, end: float, decay_steps: int) -> float:
    """Linear epsilon schedule from ``start`` -> ``end`` over ``decay_steps`` steps."""

    if decay_steps <= 0:
        return end
    frac = min(max(step / decay_steps, 0.0), 1.0)
    return float(start + (end - start) * frac)


class DQNAgent:
    """Encapsulates online + target networks, replay, and a single optimization step."""

    def __init__(
        self,
        online: DQN,
        target: DQN,
        *,
        num_actions: int,
        gamma: float,
        learning_rate: float,
        replay_capacity: int,
        double: bool,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.online = online.to(self.device)
        self.target = target.to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad = False

        self.num_actions = int(num_actions)
        self.gamma = float(gamma)
        self.double = bool(double)

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=float(learning_rate))
        self._py_rng = random.Random(seed)
        self.replay = ReplayBuffer(replay_capacity, rng=self._py_rng)

    @staticmethod
    def _to_float_tensor(frames: Iterable[np.ndarray], device: torch.device) -> torch.Tensor:
        arr = np.stack(list(frames), axis=0).astype(np.float32) / 255.0
        return torch.from_numpy(arr).to(device)

    def select_action(self, obs: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy: explore uniformly with probability ``epsilon``."""

        if self._py_rng.random() < epsilon:
            return self._py_rng.randrange(self.num_actions)
        with torch.no_grad():
            x = torch.from_numpy(obs.astype(np.float32) / 255.0).unsqueeze(0).to(self.device)
            q = self.online(x)
            return int(q.argmax(dim=1).item())

    def remember(self, t: Transition) -> None:
        self.replay.push(t)

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def can_train(self, batch_size: int) -> bool:
        return len(self.replay) >= batch_size

    def train_step(self, batch_size: int) -> float:
        """Sample a batch and run one optimization step. Returns scalar loss."""

        batch = self.replay.sample(batch_size)
        obs_t = self._to_float_tensor((b.obs for b in batch), self.device)
        next_obs_t = self._to_float_tensor((b.next_obs for b in batch), self.device)
        actions_t = torch.tensor([b.action for b in batch], dtype=torch.long, device=self.device)
        rewards_t = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=self.device)
        done_t = torch.tensor([float(b.done) for b in batch], dtype=torch.float32, device=self.device)

        q_pred = self.online(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double:
                next_actions = self.online(next_obs_t).argmax(dim=1, keepdim=True)
                next_q = self.target(next_obs_t).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target(next_obs_t).max(dim=1).values
            target = rewards_t + self.gamma * (1.0 - done_t) * next_q

        loss = F.smooth_l1_loss(q_pred, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())
