# Universal Retro-DQN (console): implementation ExecPlan

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

The repository root for git and shared policy is `gameAI/`. This plan applies to the `console/` subtree. The global planning rules live in `PLANS.md` at the repository root (`gameAI/PLANS.md`). Maintain this document in accordance with that file.

## Purpose / Big Picture

After this work, a developer with a valid stable-retro installation and imported ROMs can point the program at a single TOML run file, train a DQN-family agent on a named game, and obtain a timestamped checkpoint on disk. They can run a short automated test suite that does not require proprietary ROM data where tests are written to skip or mock appropriately. A second developer can reproduce the behavior from this document and the committed code alone.

The user-visible proof is: run the test command and see passing tests; run the training command with a real game id and see the process print periodic loss or episode return and write a checkpoint file whose name encodes the game id, whether double Q-learning and dueling heads were enabled, and a UTC timestamp to second precision.

## Progress

- [x] (2026-05-07) Author initial ExecPlan and decision log from agreed requirements.
- [x] (2026-05-15) Align README and ExecPlan: lock in `pyproject.toml` (drop `requirements.txt`), enumerate `src/wrappers/` subpackage in Milestone A, fix the reward extension surface to a single subclass-only convention, add a concrete example TOML, and add `console/.gitignore` covering `models/`, `random_text.py`, and `stable_retro_setup_notes.txt`.
- [x] (2026-05-15 22:31Z) Scaffold `console/` Python package layout: `src/__init__.py`, `src/wrappers/__init__.py`, `src/{config,environment,network,agent,train,play}.py`, `configs/`, `configs/rewards/`, `models/` (gitignored), `tests/`, `pyproject.toml` (single dependency declaration; setuptools `find` over `src*`, plus `py-modules = ["main"]`), and `main.py` CLI module.
- [x] (2026-05-15 22:35Z) Implement TOML run config loader and validation (`src/config.py`): typed `RunConfig`/`TrainingConfig` dataclasses, `tomllib`/`tomli` shim for Python 3.10+, `ConfigError` with key-naming messages, defaults for `double`/`dueling`/`frame_stack`/`reward_shaping`, `target_res` validation, "shaping requires module" check.
- [x] (2026-05-15 22:38Z) Implement environment factory (`src/environment.py`): `retro.make(..., render_mode=None)` so the default `human` viewer never spawns during training, `Discretizer` -> `VisionWrapper(target_res?)` -> optional `Wrapper` from `configs/rewards/<name>.py`. Pyglet/Cocoa workaround installed at module import on darwin (idempotent).
- [x] (2026-05-15 22:40Z) Implement `src/wrappers/`: `discretizer.py` (NOOP + per-button singletons by default; per-game `combos=` overridable), `vision.py` (BT.601 grayscale + optional cv2 resize with NumPy fallback + frame stack -> channel-first uint8), `reward_wrapper.py` (thin `RewardWrapper` alias over `gymnasium.RewardWrapper`).
- [x] (2026-05-15 22:42Z) Implement `src/network.py` (DQN-Nature CNN; flattened linear input from a dummy forward pass; optional dueling head with mean-zero advantage aggregation) and `src/agent.py` (replay buffer, linear epsilon schedule, vanilla/Double targets, gradient clipping).
- [x] (2026-05-15 22:44Z) Implement `src/train.py` (training loop, periodic stdout logging, target-net sync, checkpoint write under `models/`) and `src/play.py` (greedy + small epsilon rollout that loads `online_state_dict`). Pure-function `make_checkpoint_stem` and `game_slug` helpers expose the naming rule for tests and downstream tooling.
- [x] (2026-05-15 22:46Z) Implement `main.py` CLI: `train` and `play` subcommands; `--seed`/`--device`/`--max-steps` overrides applied by `dataclasses.replace` so the parsed `RunConfig` stays immutable.
- [x] (2026-05-15 22:48Z) Author `console/configs/example_run.toml` (verbatim from "Example run TOML") and `console/configs/airstriker_smoke.toml` for the validation transcript. A minimal example reward module lives at `configs/rewards/example_scaling.py`.
- [x] (2026-05-15 22:50Z) Add pytest suite under `console/tests/`: `test_config.py`, `test_checkpoint_naming.py`, `test_network.py`, `test_agent.py`, `test_wrappers.py`, `test_environment.py` (ROM-gated), `test_cli.py`. ROM-gated test skips when `UNIVERSAL_RETRO_DQN_TEST_GAME` is unset.
- [x] (2026-05-15 22:52Z) Install editable (`pip install -e .` in `console/` against the reference `retro_env` env) and run validation: `pytest -q` -> `64 passed, 1 skipped`; with `UNIVERSAL_RETRO_DQN_TEST_GAME=Airstriker-Genesis-v0` -> `65 passed`. Smoke training run produced `models/Airstriker-Genesis-v0_d1_u1_v20260516T054130Z.pt`; `play` rollout reported `total_reward=20.0000`.

## Surprises & Discoveries

Document unexpected behaviors during implementation with short evidence.

- Observation: On Apple Silicon (arm64) macOS, `pyglet` always selects `CocoaAlternateEventLoop` regardless of the `osx_alt_loop` flag, and its `exit()` expects `run()` to have populated `self.platform_event_loop`. A gym-style usage that only opens a window (no `run()`) crashes `env.close()`. The workaround is to lazily bind the global platform event loop in `CocoaAlternateEventLoop.exit`. Stable-retro's `SimpleImageViewer` also caps render width at 500 px and exposes no knob, so a subclass override is needed for usable resolution.
  Evidence: Local exploration script (gitignored as `console/random_text.py`) reproduces both issues and demonstrates the monkey-patch fix. The vision wrapper (or, if we use stable-retro's built-in viewer at all, the render path in `play.py`) must include or replace this workaround on macOS arm64.

- Observation (2026-05-15): `stable-retro` 1.0.0's `RetroEnv.__init__` defaults `render_mode="human"`, so even a "headless" use (only `reset` + `step` + `close`) auto-spawns the pyglet viewer inside `step()` via the implicit `if self.render_mode == "human": self.render()` branch. On macOS arm64 this triggers the `CocoaAlternateEventLoop.platform_event_loop` crash above on `env.close()`.
  Evidence: First end-to-end attempt of `tests/test_environment.py::test_make_env_builds_full_stack_when_game_available` failed with `AttributeError: 'CocoaAlternateEventLoop' object has no attribute 'platform_event_loop'` even though the test never called `env.render()`. The fix in `console/src/environment.py::make_env` is twofold: (a) install the pyglet/Cocoa workaround on `darwin` at module import (idempotent, marked via `cls._console_patched`), and (b) pass `render_mode=None` to `stable_retro.make` from training paths so the viewer never spawns at all. `play(..., render=True)` opts back into `render_mode="human"` because the patch makes the close path safe.

- Observation (2026-05-15): The canonical DQN-Nature CNN (kernels 8/4/3 with strides 4/2/1) requires at least ~64 px on each spatial axis; smaller inputs (e.g. 32x32 used in an early draft of `tests/test_agent.py`) collapse to a 2x2 feature map and crash with `Calculated padded input size per channel: (2 x 2). Kernel size: (3 x 3). Kernel size can't be greater than actual input size`. Tests now use the canonical 84x84 shape, which also matches Atari-style preprocessing.
  Evidence: First `pytest -q` run after scaffolding reported six failures all from `RuntimeError: Calculated padded input size per channel: (2 x 2). Kernel size: (3 x 3). Kernel size can't be greater than actual input size` originating in `torch.nn.Conv2d`. Switching `_TEST_OBS_SHAPE` to `(4, 84, 84)` made the suite green: `64 passed, 1 skipped`.

- Observation (2026-05-15): `torch.load` in PyTorch 2.5.1 emits a `FutureWarning` recommending `weights_only=True`. Because our checkpoint payload deliberately bundles the run config (a Python dict) alongside the state dicts, we have to keep `weights_only=False`; doing so explicitly silences the warning. The risk is acceptable because the file is one we wrote ourselves to `console/models/`.
  Evidence: First `play` invocation printed the `FutureWarning` from `src/play.py:torch.load`. Adding `weights_only=False` removed the warning while preserving the loadable config metadata.

## Decision Log

- Decision: Use two independent booleans `double` and `dueling`, default `false`, instead of a single `dqn_type` enum.
  Rationale: Double Q-learning changes the bootstrap target computation; dueling changes the network head. They compose naturally (dueling architecture with either vanilla or double targets).
  Date/Author: 2026-05-07 / project owner via agent.

- Decision: One TOML file per training or play run; no separate defaults merge in v1.
  Rationale: Simplifies reproducibility and agent autonomy; all hyperparameters for a run live in one artifact-friendly file.
  Date/Author: 2026-05-07 / project owner via agent.

- Decision: Checkpoint stem format `{game_slug}_d{0|1}_u{0|1}_v{YYYYMMDDTHHMMSSZ}.pt` where `d` reflects `double`, `u` reflects `dueling` (mnemonic: dueling “u” flag), and the timestamp is UTC with second precision, for example `20260507T143022Z`.
  Rationale: Filenames stay sortable and collision-resistant across toggles; matches the agreed pattern extending `{game}_d{0|1}_v{timestamp}` with an explicit dueling bit.
  Date/Author: 2026-05-07 / project owner via agent.

- Decision: Spatial observation shape follows the environment’s built-in shape after stable-retro construction and standard preprocessing that does not change height and width (for example grayscale only changes channels). An optional TOML resize overrides spatial dimensions when present.
  Rationale: ROM and core define native raster dimensions; the program must not invent width and height. A manual override remains for experimentation or edge cases.
  Date/Author: 2026-05-07 / project owner via agent.

- Decision: Per-game reward shaping lives as Python modules under `console/configs/rewards/`, each providing a small `RewardWrapper` subclass or factory discoverable from the run TOML.
  Rationale: Keeps game-specific logic out of core `src/` while remaining versioned beside run configs.
  Date/Author: 2026-05-07 / project owner via agent.

- Decision: Primary development and CLI execution assume current working directory `console/` unless otherwise noted.
  Rationale: Matches repository layout and keeps paths short in commands.
  Date/Author: 2026-05-07 / project owner via agent.

- Decision: Use `pyproject.toml` as the single source of dependency declarations; do not also ship a `requirements.txt`.
  Rationale: One artifact-friendly file aligns with the "one TOML file per run" ethos already adopted for run configs, avoids drift between two dependency files, and is the modern Python packaging default. A lock file (e.g. `uv.lock` or `pip`-produced lock) may be added later if reproducibility from a clean machine demands it; until then the developer pins exact versions in `pyproject.toml` at implementation time.
  Date/Author: 2026-05-15 / project owner via agent.

- Decision: Introduce `console/src/wrappers/` as a Python subpackage containing exactly three modules at v1: `discretizer.py` (the `MultiBinary` to `Discrete` reducer), `vision.py` (grayscale, channel-order conventions, optional `target_res` resize, frame stack), and `reward_wrapper.py` (the `RewardWrapper` base class).
  Rationale: The original Milestone A skeleton enumerated only `src/{environment,network,agent,train,play}.py` while the Interfaces section already referenced `console/src/wrappers/reward_wrapper.py`. Making the subpackage explicit removes that ambiguity and gives Milestone B a concrete answer to "where does each wrapper live".
  Date/Author: 2026-05-15 / project owner via agent.

- Decision: Per-game reward modules under `console/configs/rewards/` must expose exactly one class named `Wrapper` that subclasses `RewardWrapper` from `console/src/wrappers/reward_wrapper.py`. No factory function (`build_reward_wrapper(env)`) and no introspection-based discovery in v1.
  Rationale: A fixed conventional class name makes the loader trivial (one line: `mod.Wrapper(env)`) while keeping per-game modules as short as possible (one class, no factory boilerplate). This matches the Gymnasium / stable-retro wrapper pattern users already know and follows the AGENTS.md guidance of "transparency over cleverness". If a future need arises to compose multiple wrappers, the `Wrapper` subclass can do the composition inside its own `__init__`.
  Date/Author: 2026-05-15 / project owner via agent.

- Decision: `console/.gitignore` is committed and ignores (a) the `models/` output directory, (b) `random_text.py` (an exploration/smoke script for stable-retro's render path on Apple Silicon, kept locally), and (c) `stable_retro_setup_notes.txt` (developer-local install notes).
  Rationale: The two files predate this plan and are not part of the production surface; per AGENTS.md, experimental code and local notes should not mix with production code. Ignoring them keeps `git status` clean without forcing immediate deletion, and the pyglet/Cocoa workaround observed in `random_text.py` is captured in `Surprises & Discoveries` so the lesson is not lost.
  Date/Author: 2026-05-15 / project owner via agent.

- Decision: The default discretizer maps `Discrete(1 + len(buttons))` to `[NOOP] + [single-button]`. Per-game combos (e.g. classic Sonic curated combos) can be supplied later via the `combos=` parameter on `Discretizer`, but no v1 TOML key exposes this.
  Rationale: A "universal" default must work for any imported ROM with no per-game config. NOOP plus singletons preserves all single-button actions while keeping the discrete space small enough for DQN. Per-game tuning is a future ExecPlan revision; the plumbing is already in place via the constructor argument.
  Date/Author: 2026-05-15 / project owner via agent.

- Decision: `src/environment.make_env` defaults to `render_mode=None` (instead of stable-retro's `"human"` default) and the pyglet/Cocoa workaround is installed unconditionally at `src/environment.py` import on darwin.
  Rationale: stable-retro 1.0.0 auto-renders inside `step()` whenever `render_mode == "human"`, which is the default. On macOS arm64 the resulting viewer crashes `env.close()` (see Surprises). Defaulting to `None` keeps training headless and reproducible; `play(render=True)` opts back into the human viewer because the patched `close` path is now safe.
  Date/Author: 2026-05-15 / project owner via agent.

- Decision: Checkpoints serialize a `dict` containing both state dicts plus the salient `RunConfig` fields (`game`, `state`, `double`, `dueling`, `frame_stack`, `target_res`). `torch.load` is therefore called with `weights_only=False` in `src/play.py`.
  Rationale: Bundling the architectural shape information makes a checkpoint self-describing enough to bootstrap `play.py` without the run TOML. The pickle surface is acceptable because the files originate from our own training runs in `console/models/`.
  Date/Author: 2026-05-15 / project owner via agent.

- Decision: ROM-dependent integration testing is gated by the env var `UNIVERSAL_RETRO_DQN_TEST_GAME` (named after an imported game id, e.g. `Airstriker-Genesis-v0`). Without it, `tests/test_environment.py::test_make_env_builds_full_stack_when_game_available` skips.
  Rationale: AGENTS.md requires deterministic, fast tests that work on a clean machine. ROM-bearing tests can still run by setting the env var, satisfying the "Validation and Acceptance" minimum bar.
  Date/Author: 2026-05-15 / project owner via agent.

## Outcomes & Retrospective

What shipped (2026-05-15 22:55Z):

* Full `console/` Python package implementing every milestone (A-E) end-to-end. Single-file run TOMLs drive a stable-retro env, a configurable DQN (vanilla or Double, with or without dueling), and produce timestamped checkpoints under `console/models/` that round-trip through `play.py`.
* `pytest -q` from `console/` reports `64 passed, 1 skipped` on a machine without `UNIVERSAL_RETRO_DQN_TEST_GAME` set; with that env var pointing at any imported game, all `65 passed`. Test coverage spans config schema (typed loader + every required-key error path), checkpoint stem format (UTC second-precision; flag bits), discretizer / vision / reward wrappers (against a fake retro-shaped env), DQN forward shape across multiple input sizes and dueling variants, agent train-step parameter updates, ε-greedy bounds, target-net sync, and the reward-module loader's three error modes.
* End-to-end smoke transcript on the reference environment (`retro_env` miniforge env, Apple Silicon, Python 3.10.20, torch 2.5.1, stable-retro 1.0.0):

      $ python main.py train --config configs/airstriker_smoke.toml --log-every 100
      [step 100/500] eps=0.550 loss=0.0000 ep=0 fps=44.8
      [step 200/500] eps=0.100 loss=0.0000 ep=0 fps=44.0
      [step 300/500] eps=0.100 loss=0.0000 ep=0 fps=43.5
      [step 400/500] eps=0.100 loss=0.0000 ep=0 fps=43.2
      [step 500/500] eps=0.100 loss=0.0001 ep=0 fps=43.2
      [done] checkpoint saved to .../console/models/Airstriker-Genesis-v0_d1_u1_v20260516T054130Z.pt

      $ python main.py play --config configs/airstriker_smoke.toml \
          --checkpoint models/Airstriker-Genesis-v0_d1_u1_v20260516T054130Z.pt \
          --max-steps 200 --epsilon 0.1
      total_reward=20.0000

  The filename matches the contracted format `{game_slug}_d{0|1}_u{0|1}_v{YYYYMMDDTHHMMSSZ}.pt`.

What was deferred (intentional v1 omissions):

* Per-game discretizer combo tables. The plumbing exists (`Discretizer(env, combos=...)`) but no TOML key surfaces it yet. Add when the first per-game tuning needs it.
* Resume-from-checkpoint training. `train.py` always starts from random weights; resume semantics are flagged in the Idempotence section but left for v2.
* CSV / TensorBoard logging. The training loop prints scalar lines to stdout instead. Easy to plug into `train.py` later if richer dashboards are wanted.
* Lock file (`uv.lock` / `pip` lock). `pyproject.toml` is the declared source of truth; revisit if reproducibility from a clean machine becomes painful.

Lessons learned:

* The biggest concrete trap was stable-retro 1.0.0's implicit `render_mode="human"` default; an early integration test failed on macOS arm64 not because of any DQN code but because `env.close()` synchronously crashed pyglet's `CocoaAlternateEventLoop`. The fix (workaround patch + explicit `render_mode=None`) is small but the discovery cost was real, and it's now codified in two places: `_install_pyglet_cocoa_workaround()` runs at `src/environment.py` import time, and `make_env` always passes `render_mode=None` from the training path. The same lesson is captured in two Surprises entries plus a Decision Log entry so a future contributor can't accidentally re-introduce it.
* Pinning the network shape tests to 84x84 (rather than a smaller tensor used purely for "fast unit testing") matters: the canonical DQN-Nature CNN does not survive arbitrarily small inputs. Aligning with Atari's real preprocessing made the tests both faster *and* more representative.
* "Plan before coding" actually paid off here. The Decision Log decisions for the checkpoint stem rule, the `Wrapper` class convention, and the `pyproject.toml`-only dependency story turned what could have been a rambling implementation into a sequence of isolated mechanical edits that landed first-try (modulo the two Surprises above).

## Context and Orientation

**Reinforcement learning in one sentence:** the program tries actions in a game, receives rewards and screen pixels, and adjusts neural network weights so that actions that lead to higher long-term reward are preferred.

**Deep Q-network (DQN) in one sentence:** a convolutional neural network maps a stack of recent frames to a vector of estimated future discounted return for each discrete action; the agent mostly picks the action with the highest estimate and sometimes explores randomly.

**stable-retro** is a library that loads classic console games as environments similar to Gymnasium environments. You pass a string game id (for example `SonicTheHedgehog-Genesis`). The observation is typically an image tensor or array; the action space is often a `MultiBinary` or similar space that must be reduced to a finite discrete set for standard DQN.

**This repository slice:** all new code for this effort lives under `console/` as described in the Plan of Work. The parent `gameAI` repo may contain unrelated experiments; do not depend on sibling folders for imports unless this ExecPlan is later updated to say otherwise.

**Terminology:** “Vanilla” here means `double = false` and standard DQN bootstrap targets using the target network’s maximum over actions. “Double” means `double = true` and the Double DQN target that selects the greedy action with the online network but evaluates it with the target network. “Dueling” means `dueling = true` and the network uses a value stream plus an advantage stream combined with the dueling aggregation formula.

## Plan of Work

Tell the story in milestones. Each milestone ends with something runnable and a proof.

**Milestone A — Skeleton and configuration.** Create the directory tree under `console/`: `src/` with top-level modules `environment.py`, `network.py`, `agent.py`, `train.py`, `play.py`, plus a `src/wrappers/` subpackage containing `discretizer.py` (the `MultiBinary` → `Discrete` reducer), `vision.py` (grayscale, channel-order conventions, optional `target_res` resize, frame stack), and `reward_wrapper.py` (the `RewardWrapper` base class). Place `main.py` at the `console/` root as the CLI entry point. Add `configs/rewards/` for optional per-game reward code, `models/` as a gitignored output directory, `tests/` for pytest, and `console/.gitignore` to cover `models/` plus the two local exploration files noted in the Decision Log. Use `pyproject.toml` (no `requirements.txt`) as the single dependency declaration; the pins must be sufficient for a clean machine to install PyTorch, stable-retro, a Gymnasium-compatible stack, OpenCV for resizing, and a TOML parser (`tomllib` on Python 3.11+ or `tomli` on 3.10). Implement a typed or documented loader for a single run TOML that includes at least: `game` string, optional `state` string passed through to `retro.make`, `double` and `dueling` booleans defaulting false, `frame_stack` integer default 4, boolean `reward_shaping`, optional `reward_module` string naming a file under `configs/rewards/` (e.g. `reward_module = "sonic_custom"` resolves to `configs/rewards/sonic_custom.py`), optional `target_res` as a pair `[height, width]` for the resize backup, and a `[training]` table with the hyperparameters the training loop needs (learning rate, gamma, batch size, replay capacity, epsilon schedule endpoints, total steps or episodes, target network update period, seed, device). A concrete worked example is given in the "Example run TOML" section below. Proof for this milestone: import the config module in pytest and validate that missing required keys raise a clear error.

**Milestone B — Environment pipeline.** Implement `src/environment.py` as the orchestrator: it calls `retro.make(game=..., state=...)`, then composes the wrappers from `src/wrappers/` in this fixed order: discretizer (`wrappers.discretizer`) that exposes `Discrete(n)` for DQN, vision stack (`wrappers.vision`) which applies grayscale and channel-order conventions consistent with the network (document the convention in code comments near the wrapper), an optional resize when `target_res` is set in TOML, and then frame stacking. The spatial shape used for the CNN must be measured after these wrappers. If `reward_shaping` is true, the orchestrator imports the module referenced by `reward_module` from `console/configs/rewards/` and wraps the env with that module's `Wrapper` class (a `RewardWrapper` subclass — see "Reward extension interface" under Interfaces and Dependencies). Proof: a unit test builds the env stack with `reward_shaping` false and asserts observation shape dtype and action space type without requiring a licensed ROM if you can use a tiny public domain integration; if that is not feasible, gate the test behind an environment variable and document it in Concrete Steps.

**Milestone C — Network and agent.** Implement `network.py` with a convolutional torso whose flattened feature size is computed from a dummy forward pass on a tensor shaped like the post-wrapper observation, so changing games does not break linear layers. Implement dueling heads behind the `dueling` flag with the aggregation \(Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))\). Implement `agent.py` with experience replay, epsilon-greedy action selection, optimizer step, and TD loss with vanilla vs double targets controlled by `double`. Proof: unit tests on random tensors that the network output has shape `[batch, num_actions]` and that a single optimization step runs without error on fake batches.

**Milestone D — Training, checkpoints, play.** Implement `train.py` to run the loop, log scalar summaries to stdout or a small CSV in `models/logs/` (optional), and save checkpoints under `models/` using the stem rule in the Decision Log. Timestamp generation must use UTC rounded to whole seconds in the filename suffix `v{YYYYMMDDTHHMMSSZ}`. Implement `play.py` to load a checkpoint and run the policy greedily or with small epsilon, rendering optional. Proof: with a user-provided ROM id, a short run writes a file matching the stem pattern.

**Milestone E — Tests and docs touch-up.** Add pytest tests for config parsing, network shape logic, and agent tensor paths. Add a minimal README note under `console/README.md` stating the canonical test command (one line is enough; detailed behavior stays in this ExecPlan). Proof: pytest exits zero from `console/`.

## Concrete Steps

Assume POSIX shell. Replace example game ids with ones present in the developer’s stable-retro import folder.

Working directory for application commands: `console/`.

Create and activate a virtual environment at the developer’s discretion, then install the project (including dependencies pinned in `console/pyproject.toml`) in editable mode from `console/`:

    pip install -e .

Install or build stable-retro according to upstream documentation. On Apple Silicon, follow Farama’s stable-retro macOS guidance; local notes in `console/stable_retro_setup_notes.txt` may help but do not override upstream if they conflict.

Run the full unit suite from `console/`:

    pytest -q

Expected: all tests pass; tests that need ROMs print `s` for skipped when ROM or env vars are absent, never fail the suite for missing local data.

Run a short training smoke (requires a valid imported game id):

    python main.py train --config configs/example_run.toml

Expected: process starts without shape errors, prints periodic summaries, and creates a file under `models/` whose name matches `{game_slug}_d{0|1}_u{0|1}_v{YYYYMMDDTHHMMSSZ}.pt`.

Run inference smoke (exact flags to be implemented, illustrative):

    python main.py play --config configs/example_run.toml --checkpoint models/<stem_from_training>.pt

Expected: runs without exception; if render is enabled and a display exists, a window may open depending on stable-retro backend.

## Validation and Acceptance

Acceptance is behavioral. A novice should be able to clone the repo, open `console/EXECPLAN_UNIVERSAL_RETRO_DQN.md`, follow Concrete Steps, and observe passing pytest when skips are expected, plus a training run that writes a correctly named checkpoint when a ROM is available.

Minimum automated bar: `pytest -q` from `console/` passes on CI or a fresh laptop without ROM-specific tests failing when ROMs are absent (use `pytest.importorskip` or environment gating).

Manual bar: one training run on a chosen game shows non-trivial learning signal (for example average episode score trending up or loss decreasing) over a modest step budget; exact thresholds are intentionally not fixed in v1 to avoid brittle ExecPlan edits.

## Idempotence and Recovery

Training runs are additive: each run writes a new timestamped checkpoint and does not overwrite prior stems unless the operator deletes files. Re-running the same TOML in the same second on the same machine could theoretically collide; if that becomes an issue, a future revision may append a short random suffix. Recovery from partial runs is simply to start a new run or resume if a future version adds resume semantics (not required in v1 unless already trivial).

## Artifacts and Notes

Successful end-to-end transcript (2026-05-15 22:55Z; reference env: miniforge `retro_env`, Python 3.10.20, torch 2.5.1, stable-retro 1.0.0):

    $ pytest -q
    ....................................s............................        [100%]
    SKIPPED [1] tests/test_environment.py:61: set UNIVERSAL_RETRO_DQN_TEST_GAME=<imported_game_id> to exercise this test
    64 passed, 1 skipped in 1.47s

    $ UNIVERSAL_RETRO_DQN_TEST_GAME=Airstriker-Genesis-v0 pytest -q
    .................................................................        [100%]
    65 passed in 1.53s

    $ python main.py train --config configs/airstriker_smoke.toml --log-every 100
    [step 100/500] eps=0.550 loss=0.0000 ep=0 fps=44.8
    [step 200/500] eps=0.100 loss=0.0000 ep=0 fps=44.0
    [step 300/500] eps=0.100 loss=0.0000 ep=0 fps=43.5
    [step 400/500] eps=0.100 loss=0.0000 ep=0 fps=43.2
    [step 500/500] eps=0.100 loss=0.0001 ep=0 fps=43.2
    [done] checkpoint saved to .../console/models/Airstriker-Genesis-v0_d1_u1_v20260516T054130Z.pt

    $ python main.py play --config configs/airstriker_smoke.toml \
        --checkpoint models/Airstriker-Genesis-v0_d1_u1_v20260516T054130Z.pt \
        --max-steps 200 --epsilon 0.1
    total_reward=20.0000

`configs/airstriker_smoke.toml` is the smoke run's config; it deliberately uses `total_steps = 500`, `replay_capacity = 1000`, and a small `target_res = [84, 84]` so the loop completes in seconds. `configs/example_run.toml` is the canonical schema example carried verbatim from the "Example run TOML" section.

## Example run TOML

The following block is the canonical example. The loader must accept this file verbatim (with the developer's chosen `game`/`state` substituted) and produce a fully-specified training run. Save it as `console/configs/example_run.toml`. Defaults are shown explicitly even when they match the loader's defaults so the file doubles as documentation of every available key. Optional keys are commented out.

    # console/configs/example_run.toml
    # Universal Retro-DQN run config (v1). One file per training or play session.

    game          = "Airstriker-Genesis"   # stable-retro game id (must be imported)
    state         = "Level1"               # optional; passed to retro.make(state=...)
    double        = false                  # Double DQN target (default false)
    dueling       = false                  # dueling head      (default false)
    frame_stack   = 4                      # frames per observation stack (default 4)

    reward_shaping = false                 # if true, load reward_module below
    # reward_module = "sonic_custom"       # -> console/configs/rewards/sonic_custom.py
                                           #    that file must define `class Wrapper(RewardWrapper)`

    # target_res = [84, 84]                # optional [height, width] resize after vision pipeline;
                                           # omit to use the env's built-in spatial shape

    [training]
    seed                 = 0
    device               = "cpu"           # "cpu", "cuda", or "mps"
    total_steps          = 1000000
    learning_rate        = 1e-4
    gamma                = 0.99
    batch_size           = 32
    replay_capacity      = 100000
    target_update_period = 1000            # steps between target-net syncs
    epsilon_start        = 1.0
    epsilon_end          = 0.05
    epsilon_decay_steps  = 250000

Loader expectations for the above file: every key shown without a leading `#` is read; absence of a required key (everything outside the `[training]` table except `state`, `reward_module`, and `target_res`, and every key inside `[training]`) is an error with a message naming the missing key. `frame_stack` and the booleans fall back to their defaults if absent. `reward_module` is only consulted when `reward_shaping = true`; a `true` flag without a `reward_module` is an error.

## Interfaces and Dependencies

**Python:** 3.10 or newer. Prefer 3.11+ so the standard library `tomllib` parses TOML without an extra dependency; if supporting 3.10, add `tomli` and branch in loader code.

**Core libraries:** `torch`, `stable-retro`, `gymnasium`, `numpy`, `opencv-python` (resize), `pytest` for tests. Pin versions in `console/pyproject.toml` (under `[project] dependencies`) at implementation time to the versions verified on a reference machine. There is no `requirements.txt`; if a lock file becomes necessary for reproducibility, add one (`uv.lock` or pip's lock format) but keep `pyproject.toml` as the single declared source of truth.

**Game ROMs:** stable-retro requires legally obtained ROMs imported through its integration tooling. The ExecPlan cannot bundle ROMs. Tests must not assume specific commercial games unless skipped.

**Reward extension interface:** Each optional module under `console/configs/rewards/` is referenced by a string key in the run TOML, for example `reward_module = "sonic_custom"`, mapping to file `console/configs/rewards/sonic_custom.py`. That module must define exactly one public class named `Wrapper` that subclasses `RewardWrapper` from `console/src/wrappers/reward_wrapper.py`. The loader imports the module by file path (using `importlib.util.spec_from_file_location` so `configs/rewards` does not need to be a Python package on `PYTHONPATH`, while still being careful with `sys.path` to avoid shadowing stdlib modules) and instantiates the wrapper as `mod.Wrapper(env)`. No factory function and no introspection-based discovery is supported in v1: the class name `Wrapper` is the convention. A minimal example reward module therefore looks like:

    # console/configs/rewards/sonic_custom.py
    from src.wrappers.reward_wrapper import RewardWrapper

    class Wrapper(RewardWrapper):
        def reward(self, reward):
            return reward * 0.1

If a future game needs to compose multiple wrappers, that composition happens inside `Wrapper.__init__` rather than by exporting additional symbols.

**CLI interface (v1 target):** `python main.py train --config <path.toml>` and `python main.py play --config <path.toml> --checkpoint <path.pt>` with optional flags for seed, device (`cpu`, `cuda`, `mps` if available), render, and max steps. Exact flag spellings are an implementation detail but must be documented in `main.py --help` and mirrored briefly here when frozen.

**Checkpoint stem helper:** Implement a pure function `make_checkpoint_stem(game_id: str, double: bool, dueling: bool, when: datetime)` that returns the string without extension, using UTC `when` truncated to seconds and formatted `YYYYMMDDTHHMMSSZ`, and a slug function that replaces path separators and illegal filename characters with underscores.

## Revision history

- 2026-05-07: Initial ExecPlan authored from README intent and follow-up decisions (booleans, TOML single file, UTC second timestamps, `console/configs/rewards/`, shape-first resolution policy, pytest mention, `console/` as root).
- 2026-05-07: Marked initial authoring progress complete; aligned `console/README.md` with TOML, booleans, `console/` tree, ExecPlan pointer, and `pytest -q` (high level only).
- 2026-05-15: Reconciled README/ExecPlan inconsistencies surfaced during review. Changes: (a) Milestone A now enumerates `src/wrappers/` with `discretizer.py`, `vision.py`, `reward_wrapper.py`, matching the README tree; (b) dependency management is `pyproject.toml` only, no `requirements.txt` — Concrete Steps install command updated to `pip install -e .`; (c) the reward extension surface is locked to a subclass-only convention (each module under `configs/rewards/` exports `class Wrapper(RewardWrapper)`), eliminating the previously-ambiguous "factory or subclass" wording; (d) added a canonical "Example run TOML" section so the README's claim that examples live in the ExecPlan is now true; (e) added `console/.gitignore` ignoring `models/`, `random_text.py`, and `stable_retro_setup_notes.txt`; (f) captured the pyglet / `CocoaAlternateEventLoop` workaround for macOS arm64 in Surprises & Discoveries so the lesson from the exploration script survives even though the script itself is now gitignored.
- 2026-05-15 (later): Implemented every milestone end-to-end and ran the validation transcript. New Decision Log entries cover (a) the default discretizer combo set, (b) `render_mode=None` default for `make_env` plus the unconditional pyglet workaround on darwin, (c) the bundled-config checkpoint payload requiring `weights_only=False` in `torch.load`, and (d) the `UNIVERSAL_RETRO_DQN_TEST_GAME` env-var gating for ROM tests. New Surprises entries cover the stable-retro 1.0.0 `render_mode="human"` default, the DQN-Nature minimum input size discovered via test failure, and the `torch.load` `FutureWarning`. Outcomes & Retrospective and Artifacts sections now contain the full validation transcript and explicit list of v1 omissions (per-game combos, resume training, CSV/TB logging, lock file).
