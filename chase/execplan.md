# Chase DQN with minimal CNN (PyTorch)

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

The repository root file [PLANS.md](../PLANS.md) defines what an ExecPlan must contain. This document must be maintained in accordance with PLANS.md. High-level goals and the module checklist appear in [README.md](README.md). Environment behavior (rewards, 512×512 observations, registration) is summarized in **README** and implemented in `env/`; training code lives under **`src/`** only.

## Purpose / Big Picture

After this work, a contributor can **train** a DQN agent on Chase using a **minimal CNN** so that, after sufficient episodes, the agent reliably moves the **green** actor toward the **blue** X target’s **center** on a square **N×N** grid where **N = 2^k** with **k ≥ 3** and compatible with the **512×512** renderer. You can **save** the policy weights to **`checkpoints/chase_agent.pth`** and **run an evaluation script** with **ε = 0** over several episodes to see that the agent **chases** the target.

You see it working by running the training entry point (see Concrete Steps), observing logged episode returns and loss, then running the evaluation script and verifying successful episodes (and optional visual inspection of frames).

## Progress

- [x] (2026-03-30) README and this ExecPlan created with phased roadmap and **N = 2^k** grid constraint for training.
- [x] Phase 1: **`src/wrappers.py`** — `ChaseObservationWrapper` (resize, CHW, **[0,1]** float32); tests in **`tests/test_wrappers.py`**.
- [x] Phase 2: **`src/model.py`** (**MinimalQNet**), **`src/memory.py`** (**ReplayBuffer**), **`make_policy_and_target`**; tests in **`tests/test_model.py`**, **`tests/test_memory.py`**.
- [x] (2026-04-08) Phase 3: **`src/dqn.py`** — exponential ε schedule, ε-greedy action, Huber Bellman loss vs target net, Adam (**lr** from config), replay batch → device tensors, hard target sync helper.
- [x] (2026-04-08) Phase 4: **`src/train.py`** — load **`config.yaml`**, **`src/config_utils.py`** validation/device, full loop, replay warmup, gradient step, hard target sync every **`target_update_every`** env steps, logging each **`log_every_episodes`**.
- [x] (2026-04-08) Phase 5: **`torch.save(policy.state_dict(), eval.checkpoint)`** at end of training; **`src/eval.py`** loads checkpoint, **`eval.epsilon`** (default **0**), **`eval.num_episodes`** (default **20**), prints per-episode return and **`terminated`**, plus summary success rate.
- [x] (2026-03-30) Single **[config.yaml](config.yaml)** with sections **`env`**, **`wrapper`**, **`training`**, **`eval`**, **`logging`**; README and ExecPlan updated.

## Surprises & Discoveries

- None yet.

## Decision Log

- Decision: For RL training, **N** (grid side length) is **always** **N = 2^k** with **k ≥ 3** (e.g. **8, 16, 32, …, 512**). Enforced in **`ChaseEnv`** via `_validate_grid_side`. Compatible with **512 % N == 0** for the renderer.

  Rationale: Requested for this project; simplifies power-of-two resize and tensor shapes.

  Date/Author: 2026-03-30 / contributor.

- Decision: Default resize inside the observation wrapper is **64×64** and is configurable via **`wrapper.resize`**. Raw env observations remain **512×512×3** before the wrapper.

  Rationale: 64 is sufficient for this simple grid-world while reducing training compute; config keeps flexibility.

  Date/Author: 2026-03-30 / contributor.

- Decision: **MinimalQNet** means a small stack of conv layers (e.g. two or three) plus a linear head to **4** Q-values; exact channel counts are implementation details but must stay small and documented in **`src/model.py`**.

  Rationale: “Minimal” CNN per README; keeps training fast and debuggable.

  Date/Author: 2026-03-30 / contributor.

- Decision: **One YAML file** [config.yaml](config.yaml) under **`chase/`** holds all tunable settings for env, wrapper, training, eval, and logging. Training and eval entry points load it by default; alternate paths via CLI are optional. **Checkpoint** save/load path defaults to **`checkpoints/chase_agent.pth`** (see **`eval`** section).

  Rationale: Reproducibility, a single diffable artifact per experiment, and alignment with the README roadmap.

  Date/Author: 2026-03-30 / contributor.

- Decision: **torchvision** is the supported stack for resize in the observation wrapper (e.g. `torchvision.transforms`); list **torch** and **torchvision** in project dependencies for training installs.

  Rationale: Explicit choice; aligns with PyTorch training stack.

  Date/Author: 2026-04-03 / contributor.

- Decision: Exponential ε schedule uses **ε = ε_end + (ε_start − ε_end) · exp(−t / τ)** with **τ = training.epsilon_decay_steps** and **t** the global environment-step counter. **Adam** uses **training.learning_rate** (default **1e-4**). Bellman backup uses **smooth_l1_loss** (Huber) between **Q(s,a)** and the one-step target with **Q_target** frozen in the target branch.

  Rationale: Matches ExecPlan “exponential decay”; Huber is standard DQN; learning rate stays config-driven.

  Date/Author: 2026-04-08 / contributor.

## Outcomes & Retrospective

- Not yet written. Summarize after Phase 5: final reward curves, failure modes, and whether eval meets the **acceptance criterion** below.

## Context and Orientation

**Chase** is implemented as `ChaseEnv` in [env/chase_env.py](env/chase_env.py) and registered as **Chase-v0**. Observations are raw **RGB uint8** of shape **(512, 512, 3)** before wrapping. Actions are **0–3** for up, down, left, right. The **Bellman** update uses a **replay buffer** of past transitions and two copies of the same CNN: **policy** (updated every step when sampling) and **target** (periodic copy from policy for stable targets).

**DQN** here means: store \((s, a, r, s', done)\), sample mini-batches, minimize squared error between \(Q(s,a)\) and \(r + \gamma \max_{a'} Q_{\text{target}}(s', a')\) when not terminal (if terminal, target is \(r\)).

No prior knowledge of this repository is assumed beyond reading [README.md](README.md) and this file.

## Configuration file

Hyperparameters for the environment, observation wrapper, training, evaluation, and logging are defined in a **single** YAML file: **[config.yaml](config.yaml)** at the root of `chase/`. It uses named **sections** (`env`, `wrapper`, `training`, `eval`, `logging`) so one file can be committed per experiment variant or copied for sweeps. The training and eval CLIs should load this file (default path **`config.yaml`**), merge optional CLI overrides if supported, and **validate** values (e.g. **N = 2^k** with **k ≥ 3**, divisibility of **512** by **N**). **Training and eval** should save/load weights from the path in **`eval.checkpoint`**, default **`checkpoints/chase_agent.pth`**. See the [README Configuration](README.md#configuration) section for the section table.

## Plan of Work

Implement **Phase 1** under **`chase/src/`** only. Add **`src/wrappers.py`** with a class inheriting `gymnasium.ObservationWrapper` that applies resize via **torchvision**, permute to **CHW**, and divide by **255.0**. Read **`wrapper.resize`** (default 64; configurable) from [config.yaml](config.yaml). Wire `gym.make` using **`env.id`**, **`env.n`**, **`env.max_episode_steps`**, and **`env.render_mode`** from the same file.

Implement **Phase 2**: **`src/model.py`** defines **MinimalQNet** with output dimension **4**. **`src/memory.py`** holds a **deque** (maxlen) or preallocated arrays; **push** and **sample** must return batches suitable for torch tensors. Buffer capacity and batch size come from **`training`** in [config.yaml](config.yaml).

Implement **Phase 3–4** in **`src/train.py`**: load [config.yaml](config.yaml); initialize nets, optimizer, loop episodes and steps, ε-greedy decay per **`training`** keys, replay warmup before training (`min_replay_size`), compute loss, backprop, **every `target_update_every` steps** copy policy weights to the target net; **`torch.save(policy_net.state_dict(), path)`** with **`path`** defaulting to **`checkpoints/chase_agent.pth`** (same as **`eval.checkpoint`**).

Implement **Phase 5**: add **`src/eval.py`** or **`src/train.py --eval`** that loads [config.yaml](config.yaml), loads weights from **`eval.checkpoint`** (**`checkpoints/chase_agent.pth`** by default), sets ε from **`eval.epsilon`**, runs **`eval.num_episodes`** episodes (default 20), prints success rate or total reward.

## Concrete Steps

Working directory: repository root or `chase/` as appropriate.

1. Install dependencies: `pip install -e ".[rl]"` from `chase/` with **PyTorch**, **torchvision**, **gymnasium**, **numpy**, **PyYAML** (see [pyproject.toml](pyproject.toml)).

2. Run unit tests or a smoke script: `python -m pytest chase/tests/ -q` after tests exist.

3. Train (example—replace with actual CLI once **`src/train.py`** exists):

        cd chase && python -m src.train --config config.yaml

   Hyperparameters come from **`config.yaml`** (sections **`env`**, **`wrapper`**, **`training`**, **`logging`**). Expect logged metrics each **`log_every_episodes`** or on a fixed step interval.

4. Evaluate:

        cd chase && python -m src.eval --config config.yaml

   Eval uses **`eval`** section (**`checkpoint`** → **`checkpoints/chase_agent.pth`** by default, **`num_episodes`** default 20, **`epsilon`**). Expect **ε = 0** by default; print or log per-episode return and terminal success.

Training: `cd chase && python -m src.train --config config.yaml`. Eval: `cd chase && python -m src.eval --config config.yaml`. Checkpoint path defaults to **`eval.checkpoint`** in the same YAML (**`checkpoints/chase_agent.pth`**).

## Validation and Acceptance

- Wrapped observation: shape **(3, H, W)**, **float32**, values in **[0, 1]** (within floating-point tolerance).
- **MinimalQNet** output shape matches batch × **4**; **argmax** over actions matches valid discrete actions.
- Training loss decreases over time in a typical run (not a hard threshold); episode return for a trained run is **higher** than for random policy on the same **N**.
- After **`torch.save`** to **`checkpoints/chase_agent.pth`**, loading weights reproduces the policy (no random init in eval).
- **Acceptance benchmark:** On a **64×64** logical grid (**`env.n: 64`**) with **`max_episode_steps: 500`**, the eval script (**ε = 0**) should achieve **goal reach (terminated at center)** in **at least 75%** of episodes over a fixed evaluation run (document seeds if flaky). Wrapper and training settings should match what you report for this benchmark.

## Idempotence and Recovery

Training can be restarted; replay buffer can be cleared. Checkpoints can be overwritten. No destructive steps beyond overwriting weight files.

## Artifacts and Notes

- Checkpoint path: **`checkpoints/chase_agent.pth`** (default in **`eval.checkpoint`**).
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

Revision (2026-04-03): Single **execplan.md**; all training code under **`src/`**; **torchvision** decision; checkpoint default **`checkpoints/chase_agent.pth`**; wrapper default **64** via config; eval default **20** episodes; acceptance benchmark **75%** on **n=64**, **500** max steps; README alignment.
