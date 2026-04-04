# Chase DQN with minimal CNN (PyTorch)

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

The repository root file [PLANS.md](../PLANS.md) defines what an ExecPlan must contain. This document must be maintained in accordance with PLANS.md. High-level goals and the module checklist appear in [README.md](README.md). Environment behavior (rewards, 512×512 observations, registration) is summarized in **README** and implemented in `env/`; training code lives under **`src/`** only.

## Purpose / Big Picture

After this work, a contributor can **train** a DQN agent on Chase using a **minimal CNN** so that, after sufficient episodes, the agent reliably moves the **green** actor toward the **blue** X target’s **center** on a square **N×N** grid where **N = 2^k** with **k ≥ 3** and compatible with the **512×512** renderer. You can **save** the policy weights to **`chase_agent.pth`** and **run an evaluation script** with **ε = 0** over several episodes to see that the agent **chases** the target.

You see it working by running the training entry point (see Concrete Steps), observing logged episode returns and loss, then running the evaluation script and verifying successful episodes (and optional visual inspection of frames).

## Progress

- [x] (2026-03-30) README and this ExecPlan created with phased roadmap and **N = 2^k** grid constraint for training.
- [ ] Phase 1: `ObservationWrapper` in **`src/wrappers.py`** (resize, permute, normalize); test or small script prints **(3, H, W)** `float32` in **[0, 1]**.
- [ ] Phase 2: **`src/model.py`** — **MinimalQNet**; **`src/memory.py`** — replay buffer; policy and target nets instantiated.
- [ ] Phase 3: ε-greedy with exponential decay; Bellman loss vs target net; Adam **lr=1e-4**.
- [ ] Phase 4: **`src/train.py`** — full training loop; target sync every **C** steps; logging.
- [ ] Phase 5: **`torch.save`** to **`chase_agent.pth`**; **`src/eval.py`** (or `python -m src.train --eval`); **5** episodes with **ε=0**.
- [x] (2026-03-30) Single **[config.yaml](config.yaml)** with sections **`env`**, **`wrapper`**, **`training`**, **`eval`**, **`logging`**; README and ExecPlan updated.

## Surprises & Discoveries

- None yet.

## Decision Log

- Decision: For RL training, **N** (grid side length) is **always** **N = 2^k** with **k ≥ 3** (e.g. **8, 16, 32, …, 512**). Enforced in **`ChaseEnv`** via `_validate_grid_side`. Compatible with **512 % N == 0** for the renderer.

  Rationale: Requested for this project; simplifies power-of-two resize and tensor shapes.

  Date/Author: 2026-03-30 / contributor.

- Decision: Default resize inside the observation wrapper is **84×84** unless profiling favors **64×64**—either is acceptable if fixed in **`src/wrappers.py`** and tests. Raw env observations remain **512×512×3** before the wrapper.

  Rationale: Common small-atari-style size; exact choice is TBD at implementation.

  Date/Author: 2026-03-30 / contributor.

- Decision: **MinimalQNet** means a small stack of conv layers (e.g. two or three) plus a linear head to **4** Q-values; exact channel counts are implementation details but must stay small and documented in **`src/model.py`**.

  Rationale: “Minimal” CNN per README; keeps training fast and debuggable.

  Date/Author: 2026-03-30 / contributor.

- Decision: **One YAML file** [config.yaml](config.yaml) under **`chase/`** holds all tunable settings for env, wrapper, training, eval, and logging. Training and eval entry points load it by default; alternate paths via CLI are optional. **Checkpoint** save/load path defaults to **`chase_agent.pth`** (see **`eval`** section).

  Rationale: Reproducibility, a single diffable artifact per experiment, and alignment with the README roadmap.

  Date/Author: 2026-03-30 / contributor.

- Decision: **torchvision** is the supported stack for resize in the observation wrapper (e.g. `torchvision.transforms`); list **torch** and **torchvision** in project dependencies for training installs.

  Rationale: Explicit choice; aligns with PyTorch training stack.

  Date/Author: 2026-04-03 / contributor.

## Outcomes & Retrospective

- Not yet written. Summarize after Phase 5: final reward curves, failure modes, and whether eval meets the **acceptance criterion** below.

## Context and Orientation

**Chase** is implemented as `ChaseEnv` in [env/chase_env.py](env/chase_env.py) and registered as **Chase-v0**. Observations are raw **RGB uint8** of shape **(512, 512, 3)** before wrapping. Actions are **0–3** for up, down, left, right. The **Bellman** update uses a **replay buffer** of past transitions and two copies of the same CNN: **policy** (updated every step when sampling) and **target** (periodic copy from policy for stable targets).

**DQN** here means: store \((s, a, r, s', done)\), sample mini-batches, minimize squared error between \(Q(s,a)\) and \(r + \gamma \max_{a'} Q_{\text{target}}(s', a')\) when not terminal (if terminal, target is \(r\)).

No prior knowledge of this repository is assumed beyond reading [README.md](README.md) and this file.

## Configuration file

Hyperparameters for the environment, observation wrapper, training, evaluation, and logging are defined in a **single** YAML file: **[config.yaml](config.yaml)** at the root of `chase/`. It uses named **sections** (`env`, `wrapper`, `training`, `eval`, `logging`) so one file can be committed per experiment variant or copied for sweeps. The training and eval CLIs should load this file (default path **`config.yaml`**), merge optional CLI overrides if supported, and **validate** values (e.g. **N = 2^k** with **k ≥ 3**, divisibility of **512** by **N**). **Training and eval** should save/load weights from the path in **`eval.checkpoint`**, default **`chase_agent.pth`**. See the [README Configuration](README.md#configuration) section for the section table.

## Plan of Work

Implement **Phase 1** under **`chase/src/`** only. Add **`src/wrappers.py`** with a class inheriting `gymnasium.ObservationWrapper` that applies resize via **torchvision**, permute to **CHW**, and divide by **255.0**. Read **`wrapper.resize`** (and any future keys) from [config.yaml](config.yaml). Wire `gym.make` using **`env.id`**, **`env.n`**, **`env.max_episode_steps`**, and **`env.render_mode`** from the same file.

Implement **Phase 2**: **`src/model.py`** defines **MinimalQNet** with output dimension **4**. **`src/memory.py`** holds a **deque** (maxlen) or preallocated arrays; **push** and **sample** must return batches suitable for torch tensors. Buffer capacity and batch size come from **`training`** in [config.yaml](config.yaml).

Implement **Phase 3–4** in **`src/train.py`**: load [config.yaml](config.yaml); initialize nets, optimizer, loop episodes and steps, ε-greedy decay per **`training`** keys, replay warmup before training (`min_replay_size`), compute loss, backprop, **every `target_update_every` steps** copy policy weights to the target net; **`torch.save(policy_net.state_dict(), path)`** with **`path`** defaulting to **`chase_agent.pth`** (same as **`eval.checkpoint`**).

Implement **Phase 5**: add **`src/eval.py`** or **`src/train.py --eval`** that loads [config.yaml](config.yaml), loads weights from **`eval.checkpoint`** (**`chase_agent.pth`** by default), sets ε from **`eval.epsilon`**, runs **`eval.num_episodes`** episodes, prints success rate or total reward.

## Concrete Steps

Working directory: repository root or `chase/` as appropriate.

1. Install dependencies: `pip install -e ".[rl]"` from `chase/` with **PyTorch**, **torchvision**, **gymnasium**, **numpy**, **PyYAML** (see [pyproject.toml](pyproject.toml)).

2. Run unit tests or a smoke script: `python -m pytest chase/tests/ -q` after tests exist.

3. Train (example—replace with actual CLI once **`src/train.py`** exists):

        cd chase && python -m src.train --config config.yaml

   Hyperparameters come from **`config.yaml`** (sections **`env`**, **`wrapper`**, **`training`**, **`logging`**). Expect logged metrics each **`log_every_episodes`** or on a fixed step interval.

4. Evaluate:

        cd chase && python -m src.eval --config config.yaml

   Eval uses **`eval`** section (**`checkpoint`** → **`chase_agent.pth`** by default, **`num_episodes`**, **`epsilon`**). Expect **ε = 0** by default; print or log per-episode return and terminal success.

Update this section with exact module paths when implementation lands.

## Validation and Acceptance

- Wrapped observation: shape **(3, H, W)**, **float32**, values in **[0, 1]** (within floating-point tolerance).
- **MinimalQNet** output shape matches batch × **4**; **argmax** over actions matches valid discrete actions.
- Training loss decreases over time in a typical run (not a hard threshold); episode return for a trained run is **higher** than for random policy on the same **N**.
- After **`torch.save`** to **`chase_agent.pth`**, loading weights reproduces the policy (no random init in eval).
- **Acceptance benchmark:** On a **64×64** logical grid (**`env.n: 64`**) with **`max_episode_steps: 500`**, the eval script (**ε = 0**) should achieve **goal reach (terminated at center)** in **at least 75%** of episodes over a fixed evaluation run (document seeds if flaky). Wrapper and training settings should match what you report for this benchmark.

## Idempotence and Recovery

Training can be restarted; replay buffer can be cleared. Checkpoints can be overwritten. No destructive steps beyond overwriting weight files.

## Artifacts and Notes

- Checkpoint filename: **`chase_agent.pth`** (default in **`eval.checkpoint`**).
- Example terminal line after successful eval: per-episode return and `terminated=True` when reaching goal.

## Interfaces and Dependencies

- **gymnasium**: `Env`, `ObservationWrapper`, `spaces.Discrete(4)`.
- **torch** / **torchvision**: `nn.Module`, `optim.Adam`, `save` / `load_state_dict`; **torchvision** transforms for resize in **`src/wrappers.py`**.
- **numpy** / **deque** (`collections`): replay storage.
- **PyYAML** (`yaml.safe_load` or equivalent): load [config.yaml](config.yaml) at startup.

**MinimalQNet** (sketch—implement in **`src/model.py`**):

- `forward(x)` with **x** shape **(B, 3, H, W)** after wrapper.
- Returns tensor **(B, 4)** Q-values.

**ReplayBuffer** (implement in **`src/memory.py`**):

- `push(obs, action, reward, next_obs, done)` with **obs** already tensor or numpy per project convention.
- `sample(batch_size)` returns batches of each field.

---

Revision history: Initial version (2026-03-30) — DQN + minimal CNN phases, module checklist, **N = 2^k** constraint, pointers to README.

Revision (2026-03-30): Documented single **[config.yaml](config.yaml)** with sections **`env`**, **`wrapper`**, **`training`**, **`eval`**, **`logging`**; Plan of Work and Concrete Steps now reference it; Decision Log entry added.

Revision (2026-04-03): Single **execplan.md**; all training code under **`src/`**; **torchvision** decision; **`chase_agent.pth`** throughout; acceptance benchmark **75%** on **n=64**, **500** max steps; README alignment.
