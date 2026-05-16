"""Pytest setup: ensure ``console/`` is on ``sys.path`` so ``import src…`` works.

Running ``pytest -q`` from ``console/`` already gives that; this conftest also
covers the case where pytest is invoked from a different working directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

_CONSOLE_ROOT = Path(__file__).resolve().parents[1]
if str(_CONSOLE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CONSOLE_ROOT))
