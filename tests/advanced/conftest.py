"""Shared pytest configuration for the advanced-layer suite.

Puts the mente package + tests/fixtures on sys.path so the suite runs in a
fresh worktree without an editable install.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for extra in (_ROOT / "src", _ROOT / "tests"):
    s = str(extra)
    if s not in sys.path:
        sys.path.insert(0, s)
