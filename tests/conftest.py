"""Shared fixtures for MENTE core tests.

Provides:
- ``tmp_root``: a temporary directory (wraps pytest's ``tmp_path``) that tests
  can use for on-disk artifacts without polluting the repo.
- ``bus``: a fresh in-process ``EventBus`` per test.
- ``event_capture``: a helper object whose ``handler`` coroutine records every
  dispatched ``Event`` into ``events``.

These fixtures are sync-only (they don't start the bus transport). Tests that
need a started bus should ``await bus.start()`` themselves so they own the
lifecycle.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the src/ layout importable without an editable install.
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mente.bus import EventBus  # noqa: E402
from tests.fixtures.core_events import EventCapture  # noqa: E402


@pytest.fixture
def tmp_root(tmp_path: Path) -> Path:
    """Return a per-test temporary directory."""
    return tmp_path


@pytest.fixture
def bus() -> EventBus:
    """Return a fresh in-process EventBus (default InProcessTransport)."""
    return EventBus()


@pytest.fixture
def event_capture() -> EventCapture:
    """Return a fresh event capture helper."""
    return EventCapture()
