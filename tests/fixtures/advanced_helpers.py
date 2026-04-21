"""Test helpers for the advanced-layer test suite.

Small, opinionated builders so each test reads as "make a Runtime/Bus and
exercise the scenario" rather than fifteen lines of setup.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aria.runtime import Runtime
from aria.tools import ToolRegistry
from aria.types import Intent, ReasonerTier, Response
from aria.world_model import WorldModel


async def make_runtime(tmp_path: Path, node_id: str = "aria.test") -> Runtime:
    """Build a Runtime rooted under `tmp_path` and start the bus."""
    rt = Runtime(root=tmp_path / "state", node_id=node_id)
    await rt.start()
    return rt


async def shutdown_runtime(rt: Runtime) -> None:
    """Release every runtime resource. Important on macOS/Windows where a
    still-open SQLite file keeps tmp_path from being cleaned up."""
    await rt.shutdown()


@dataclass
class EchoReasoner:
    """Trivial reasoner for remote-dispatch tests."""
    name: str = "echo.specialist"
    tier: ReasonerTier = "specialist"
    est_cost_ms: float = 1.0

    async def answer(self, intent: Intent, world: WorldModel, tools: ToolRegistry) -> Response:
        return Response(
            text=f"echo:{intent.text}",
            reasoner=self.name,
            tier=self.tier,
            confidence=0.9,
            cost_ms=self.est_cost_ms,
        )
