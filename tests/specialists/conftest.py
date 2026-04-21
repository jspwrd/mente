"""Test helpers for specialists/ — wire up a cheap WorldModel + ToolRegistry."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make `src/` importable when pytest is invoked from repo root without an
# install. Mirrors the behaviour of the `./aria` launcher.
_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from aria.bus import EventBus  # noqa: E402
from aria.tools import ToolRegistry  # noqa: E402
from aria.world_model import WorldModel  # noqa: E402


@pytest.fixture
def world_tools():
    bus = EventBus()
    world = WorldModel(bus=bus)
    tools = ToolRegistry()
    return world, tools
