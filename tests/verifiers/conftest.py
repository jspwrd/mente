"""Fixtures for verifier tests.

All stubs are pure-Python; no I/O, no asyncio event loop. ``WorldModel`` is
populated via its internal ``_beliefs`` dict directly so we don't need to
spin up a bus for these unit tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure src/ is on the path so ``import aria`` works when pytest is run from
# the project root (the package isn't installed in dev).
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from aria.bus import EventBus  # noqa: E402
from aria.types import Belief, Intent, Response  # noqa: E402
from aria.world_model import WorldModel  # noqa: E402


@pytest.fixture
def intent() -> Intent:
    return Intent(text="what time is it?", source="user")


@pytest.fixture
def world() -> WorldModel:
    """Empty world model. Tests that need beliefs call ``populate_world``."""
    return WorldModel(bus=EventBus())


def populate_world(world: WorldModel, beliefs: list[Belief]) -> WorldModel:
    """Synchronously insert beliefs without touching the bus."""
    for b in beliefs:
        world._beliefs[(b.entity, b.attribute)] = b
    return world


def make_response(
    text: str,
    *,
    reasoner: str = "test",
    tier: str = "fast",
    confidence: float = 0.9,
    cost_ms: float = 1.0,
    tools_used: list[str] | None = None,
) -> Response:
    return Response(
        text=text,
        reasoner=reasoner,
        tier=tier,  # type: ignore[arg-type]
        confidence=confidence,
        cost_ms=cost_ms,
        tools_used=list(tools_used or []),
    )


@pytest.fixture
def make_response_fixture():
    return make_response


@pytest.fixture
def populate_world_fixture():
    return populate_world
