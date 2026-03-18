# agents.md

## Purpose

This repository is designed for small-scale deep learning research and experimentation. Contributors (human or AI agents) are expected to prioritize clarity, reproducibility, and incremental progress.

A key requirement in this repo is the use of **ExecPlans** (execution plans) for any non-trivial work. These are defined in detail in `PLANS.md` and must be followed.

---

## Core Principles

1. **Reproducibility first**

   * All experiments should be reproducible from committed code.
   * Random seeds must be fixed where applicable.
   * Dependencies must be explicitly defined.

2. **Small, iterative changes**

   * Prefer small, reviewable commits.
   * Avoid large, monolithic PRs.

3. **Plan before coding**

   * Any meaningful change must begin with an ExecPlan.
   * Plans should be reviewed (by a human or self-reviewed) before execution.

4. **Transparency over cleverness**

   * Favor readable and explicit implementations over highly optimized or obscure code.

---

## ExecPlans (Required Workflow)

All non-trivial work must follow the ExecPlan workflow defined in `PLANS.md`.

### When an ExecPlan is required

* New features
* Model architecture changes
* Training pipeline changes
* Data processing changes
* Refactoring that impacts behavior

### Minimum ExecPlan expectations

* Clear objective
* Step-by-step execution plan
* Defined inputs/outputs
* Validation strategy
* Rollback or failure handling

### Execution rules

* Do not begin coding until the ExecPlan is complete.
* Follow the plan step-by-step.
* Update the plan if reality diverges, and document why.

---

## Project Structure (Guidelines)

```
/project_root
  /src            # Core code (models, training, utils)
  /experiments    # Experiment configs and logs
  /data           # Local-only data (gitignored)
  /notebooks      # Exploration (not production code)
  /tests          # Unit and integration tests
  PLANS.md        # ExecPlan specification
  agents.md       # This file
```

---

## Coding Standards (Python)

### General

* Python 3.10+
* Follow PEP8 (enforced via tooling)
* Use type hints where practical
* Prefer explicit over implicit behavior

### Formatting & Linting

* Use `black` for formatting
* Use `ruff` or `flake8` for linting
* Use `isort` for import ordering

### Example rules

* Max line length: 100–120
* Avoid unused imports and variables
* Avoid deeply nested logic

---

## Git Workflow

### Branching

* `main` should always be stable
* Use feature branches: `feature/<short-description>`

### Commits

* Use clear, descriptive commit messages
* Keep commits small and focused

### Pre-commit Hooks (Required)

Set up `pre-commit` with at least:

* black
* isort
* ruff/flake8
* trailing whitespace removal
* end-of-file fixer

Example setup:

```
pip install pre-commit
pre-commit install
```

---

## Data & Security Rules

### NEVER commit:

* API keys or credentials
* `.env` files
* Private datasets
* Large files (>50MB recommended threshold)

### Required safeguards

* Use `.gitignore` for:

  * `/data`
  * `/checkpoints`
  * `/outputs`
  * `.env`

* Use environment variables for secrets

* Provide `.env.example` where needed

---

## Experiment Tracking

* Each experiment should:

  * Have a clear config
  * Log parameters and results
  * Be reproducible from code

* Prefer lightweight tools (e.g., simple logging, CSV, or minimal tracking libs)

---

## Testing Expectations

* Add tests for:

  * Core utilities
  * Data transformations
  * Model components (where feasible)

* Tests should be:

  * Fast
  * Deterministic

---

## Notebooks

* Notebooks are for exploration only
* Do not rely on notebooks for production workflows
* Important results should be migrated into scripts

---

## AI Agent-Specific Rules

If you are an AI agent contributing to this repo:

1. You MUST read `PLANS.md` before starting any work.
2. You MUST create an ExecPlan before implementing changes.
3. You MUST not skip planning even for seemingly simple tasks if they affect behavior.
4. You SHOULD explain tradeoffs in your plan.
5. You MUST not introduce hidden dependencies or external calls without justification.

---

## Anti-Patterns (Avoid)

* Silent changes to model behavior
* Mixing experimental and production code
* Hardcoding paths or secrets
* Large, unexplained commits
* Skipping ExecPlans

---

## Summary

This repo enforces a **plan-first, reproducible, and disciplined workflow** for deep learning research. The combination of ExecPlans, strict git hygiene, and lightweight experimentation is intended to maximize clarity and minimize chaos.
