# Project: Universal Retro-DQN

### Overview
This project is a modular Deep Q-Network (DQN) framework built on `pytorch` and `stable-retro`. It is designed to be hardware-agnostic and console-independent, allowing users to train agents on any retro game by supplying a single **TOML** run config per training or play session.

Authoritative implementation steps, CLI flags, schema, and validation live in **`EXECPLAN_UNIVERSAL_RETRO_DQN.md`** (living document; follow `../PLANS.md`). From `console/`, run **`pytest -q`** for the automated test suite.

### Key Features
*   **Flexible DQN architecture:** **Double** Q-learning (on/off) and **dueling** heads (on/off) are independent boolean options in the run TOML (both default off → vanilla DQN targets and a single-stream head).
*   **Dynamic resolution handling:** Uses the environment’s built-in frame shape after stable-retro and standard vision preprocessing; an optional resize in TOML overrides spatial size when set.
*   **Custom reward shaping:** Optional per-game modules under `configs/rewards/`. Each module exports a single class `Wrapper` that subclasses `RewardWrapper` from `src/wrappers/reward_wrapper.py`; the loader instantiates it as `mod.Wrapper(env)` (see ExecPlan for the full convention).
*   **Checkpoint naming:** UTC timestamp to second precision, with flags encoded in the stem (see ExecPlan).

### Configuration

Use one TOML file per run. Keys, defaults, hyperparameters, and examples are defined in **`EXECPLAN_UNIVERSAL_RETRO_DQN.md`** (not duplicated here).

---

## Suggested File Structure

```text
console/
├── .gitignore              # Ignores models/, local exploration scripts, and setup notes
├── EXECPLAN_UNIVERSAL_RETRO_DQN.md
├── configs/
│   ├── rewards/            # Per-game reward modules; each exports `class Wrapper(RewardWrapper)`
│   └── *.toml              # One TOML file per train/play run (see ExecPlan for example)
├── models/                 # Output directory for weights (gitignored)
├── src/
│   ├── agent.py            # DQN agent (replay, ε-greedy, vanilla vs double targets)
│   ├── network.py          # CNN; optional dueling head
│   ├── wrappers/
│   │   ├── discretizer.py  # MultiBinary -> Discrete reducer
│   │   ├── reward_wrapper.py  # RewardWrapper base class
│   │   └── vision.py       # grayscale, optional resize, frame stack
│   ├── environment.py      # Orchestrates retro.make + wrapper composition
│   ├── train.py
│   └── play.py
├── tests/
├── main.py                 # CLI entry: `python main.py {train,play} --config ...`
└── pyproject.toml          # Single source of dependency declarations
```

---

## Design notes (non-normative)

The following formulas and behaviors are spelled out step-by-step in **`EXECPLAN_UNIVERSAL_RETRO_DQN.md`**.

*   **Environment:** `retro.make()`, discretizer for `MultiBinary` → `Discrete`, vision stack, optional resize from TOML, frame stacking; observation shape for the CNN is taken from the wrapped env (built-in spatial size first).
*   **Dueling aggregation:** \(Q(s,a) = V(s) + (A(s,a) - \frac{1}{|A|}\sum_{a'} A(s,a'))\).
*   **Double DQN target:** \(R + \gamma Q_{target}(s', \arg\max_a Q_{online}(s', a))\); vanilla DQN uses the target network for both selection and evaluation in the bootstrap max.

Per-game reward logic: subclass **`RewardWrapper`** in `configs/rewards/` (see ExecPlan).
