# Chase

A **grid-world** game: a green **actor** (one cell) and a blue **cross target** (a 3×3 X) are placed on a square **N×N** field. The actor moves **up, down, left, or right**; the goal is to reach the **center** of the target patch. The same setup supports **human play**, **scripted demos**, and **reinforcement learning** via Gymnasium.

Repository workflow: [PLANS.md](../PLANS.md) (ExecPlans), [AGENTS.md](../AGENTS.md) (contributor guidelines).


| Document                   | Role                                                                                                                 |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| [execplan.md](execplan.md) | Single project ExecPlan: Gymnasium `ChaseEnv`, DQN training with a minimal PyTorch CNN (phases below), rendering, registration |


## Configuration

Training and environment hyperparameters live in **one file**, **[config.yaml](config.yaml)**, at the root of the `chase/` package (same directory as this README). It is organized into **sections** so env, preprocessing, training, evaluation, and logging stay in one place and are easy to diff in version control.

| Section | Purpose |
| -------- | -------- |
| **`env`** | Gymnasium env id (`Chase-v0`), grid size **`n`**, `max_episode_steps`, optional `render_mode`. |
| **`wrapper`** | Observation preprocessing (e.g. **resize** for CNN input). |
| **`training`** | DQN hyperparameters: schedules, replay, optimizer, `device`, seeds, episode count. |
| **`eval`** | Checkpoint path (**`chase_agent.pth`**), number of eval episodes, **ε = 0** for greedy eval. |
| **`logging`** | How often to log metrics (e.g. every *n* episodes). |

The training entry point should accept a path (default **`config.yaml`**) and load it with a YAML parser (e.g. **PyYAML**). Implementations must **validate** constraints from this README (e.g. **N = 2^k** with **k ≥ 3**, **512 % N == 0**). For new experiments, copy **`config.yaml`** to a new file or override keys via CLI if the trainer supports it.

## Grid size constraint (training pipeline)

The playing area is **square** (**N×N**). For RL training and preprocessing in this project, **N is restricted to a power of two** with exponent at least 3:


N = 2^k \quad\text{for integer } k \ge 3


(so at least **8**). The environment’s **default** framebuffer is **512×512** RGB; that requires **512 % N == 0**, which holds for every such **N** up to **512** (e.g. 8, 16, 32, 64, 128, 256, 512). Implementations should validate **N** at env or wrapper construction.

## Environment (Gymnasium)

Summary; training and evaluation details are in [execplan.md](execplan.md).

- **Observation (default):** **512×512×3** `uint8` RGB from the environment (`ChaseEnv`). An optional **observation wrapper** may resize for the CNN (see phases below); the **raw** env observation remains **512×512×3** unless you replace rendering.
- **Agent:** one **green** cell; **actions** `Discrete(4)`: **0 = up**, **1 = down**, **2 = left**, **3 = right**.
- **Target:** **blue** X on **3×3 logical cells**; episode success at the **center** of that patch (see ExecPlan).
- **Rewards** (implemented in `env/chase_env.py`): Let \(d\) be **Manhattan distance** from the agent to the target **center** after a step, and \(d'\) the distance before that step.
  - **Reach the center:** **+1.0** on that step (episode terminates).
  - **Otherwise:** a small step cost **−0.01** each step.
  - **Move toward the target:** if \(d < d'\) (the agent strictly decreased Manhattan distance to the center), add **+0.02** shaping reward on top of the step cost (typically **+0.01** net on a “good” step). No extra bonus if \(d = d'\) (e.g. blocked by a wall) or if \(d > d'\).
- **Dependencies:** **Gymnasium**, **NumPy**; **PyTorch** and **torchvision** (for CNN training and image resize in the observation wrapper); **PyYAML** for config. See [execplan.md](execplan.md).

## Reinforcement learning roadmap (DQN + minimal CNN)

Goal: train the agent with **Deep Q-Networks (DQN)** using a **minimal convolutional network** in **PyTorch** to map preprocessed observations to Q-values over four actions.

### Phase 1: Environment and preprocessing wrapper

**Goal:** Observations are PyTorch-friendly.

- **Observation wrapper** (subclass `gymnasium.ObservationWrapper`): consumes **512×512×3** `uint8` by default, then:
  - Resize frames (e.g. **64×64** or **84×84**; pick one and document in code under `src/`).
  - Permute **(H, W, C) → (C, H, W)**.
  - Normalize pixels to **[0, 1]** (`float32`).
- **Verify:** After `reset`, printed shape **(3, H, W)** and value range consistent with normalization.

**Success:** Tensor shape **(3, H, W)**, dtype **float32**, values in **[0, 1]**.

### Phase 2: Model and replay buffer

**Goal:** Storage and a **MinimalQNet** policy.

- **CNN (`MinimalQNet`):** Convolutional trunk, flatten, linear head with output size **action_space.n** (here **4**).
- **Replay buffer:** `deque` or circular NumPy storage; `**push(state, action, reward, next_state, done)`** and `**sample(batch_size)**`.
- **Two networks:** **policy net** (action selection / training) and **target net** (bootstrap targets), same architecture.

### Phase 3: DQN logic

**Goal:** Action selection and learning update.

- **ε-greedy** exploration with **exponential decay** (e.g. ε from **1.0** toward **~0.05**).
- **Loss:** Bellman targets y_j = r_j + \gamma \max_{a'} Q_{\text{target}}(s_{j+1}, a') (masking `done` where appropriate).
- **Optimizer:** **Adam**, learning rate **1e-4** (adjust only with justification in [execplan.md](execplan.md)).

### Phase 4: Training loop

**Goal:** End-to-end training and logging.

- **Outer:** **N** episodes.
- **Inner:** step env → store transition → sample batch → optimize.
- **Target update:** every **C** steps (e.g. **1000**), copy **policy → target** weights.
- **Log:** episode return and average loss (and any other agreed metrics).

### Phase 5: Evaluation and persistence

**Goal:** Save weights and sanity-check behavior.

- **Save / load:** policy weights as **`chase_agent.pth`** (path configurable via **`eval`** / training config; default filename **`chase_agent.pth`**).
- **Eval script:** Load weights, **ε = 0**, run several episodes (e.g. **5**) with rendering or logged success to confirm the actor **chases** the target.

**Acceptance benchmark** (full criteria in [execplan.md](execplan.md)): greedy eval (**ε = 0**) should reach the goal in **≥ 75%** of episodes on a **64×64** grid with **500** max episode steps (fixed seeds if needed).

## Agent implementation checklist

All training modules live under **`src/`** (see [execplan.md](execplan.md)).


| Module            | Responsibility                                                | Done |
| ----------------- | ------------------------------------------------------------- | ---- |
| `src/wrappers.py` | Image resize, **(H,W,C)→(C,H,W)**, normalization to **[0,1]** | [ ]  |
| `src/model.py`    | **MinimalQNet** CNN + head; output dim = **4**                | [ ]  |
| `src/memory.py`   | Replay storage, **push**, **sample**                            | [ ]  |
| `src/train.py`    | DQN loop, ε-decay, optimization, target sync, logging; **reads [config.yaml](config.yaml)** | [ ]  |


## Repository layout


| Path          | Role                                                                  |
| ------------- | --------------------------------------------------------------------- |
| `config.yaml` | **Single** YAML config: **`env`**, **`wrapper`**, **`training`**, **`eval`**, **`logging`** |
| `env/`        | Gymnasium `ChaseEnv` and registration                                 |
| `src/`        | Training code (`wrappers`, `model`, `memory`, `train`)                |
| `tests/`      | Unit and integration tests                                            |
| `notebooks/`  | Exploration and demos                                                 |
| `execplan.md` | ExecPlan (environment + DQN)                                        |

