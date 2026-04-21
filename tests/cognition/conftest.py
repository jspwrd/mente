"""Cognition-scoped pytest configuration.

If the root `tests/conftest.py` from Unit 1 is absent, this file keeps the
subtree self-sufficient: puts `src/` + `tests/` on the import path and flips
pytest-asyncio into "auto" mode so we don't decorate every coroutine.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Guarantee `import aria.*` and `import fixtures.*` resolve without install.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
_TESTS = _REPO_ROOT / "tests"
for p in (_SRC, _TESTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def pytest_configure(config):  # type: ignore[no-untyped-def]
    """Force pytest-asyncio into AUTO mode for this subtree so bare `async def`
    tests run without decorators. Safe to call even if already set."""
    try:
        config.option.asyncio_mode = "auto"
    except Exception:  # pragma: no cover - defensive
        pass
