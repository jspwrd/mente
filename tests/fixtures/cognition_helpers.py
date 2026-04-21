"""Shared helpers + fixtures for cognition-layer tests."""
from __future__ import annotations

from dataclasses import dataclass

from mente.bus import EventBus
from mente.tools import ToolRegistry
from mente.types import Belief, Intent, ReasonerTier, Response
from mente.world_model import WorldModel


async def make_world(beliefs: list[Belief] | None = None) -> WorldModel:
    """Build a WorldModel with optional initial beliefs. Uses a fresh EventBus."""
    world = WorldModel(bus=EventBus())
    for b in beliefs or []:
        await world.assert_belief(b)
    return world


def register_default_tools(tools: ToolRegistry) -> dict[str, list]:
    """Register the small set of tools FastHeuristicReasoner uses, backed by
    in-memory state so we can assert side effects.

    Returns the backing state dict so tests can inspect notes/recall output.
    """
    state: dict[str, list] = {"notes": [], "hits": []}

    @tools.register(name="clock.now", description="current time", returns="str")
    async def _clock_now() -> str:
        return "2026-04-21T00:00:00Z"

    @tools.register(name="memory.note", description="note a fact", returns="str")
    async def _memory_note(fact: str) -> str:
        state["notes"].append(fact)
        return fact

    @tools.register(name="memory.recall", description="recall all notes", returns="list")
    async def _memory_recall() -> list[str]:
        return list(state["notes"])

    @tools.register(name="memory.search", description="search memory", returns="list")
    async def _memory_search(query: str, k: int = 3) -> list[dict]:
        return list(state["hits"])

    return state


@dataclass
class StubReasoner:
    """A reasoner stub that returns a preset Response. Useful for router tests."""
    name: str
    tier: ReasonerTier
    est_cost_ms: float
    preset: Response | None = None
    calls: int = 0

    async def answer(self, intent: Intent, world: WorldModel, tools: ToolRegistry) -> Response:
        self.calls += 1
        if self.preset is not None:
            return self.preset
        return Response(
            text=f"{self.name}:{intent.text}",
            reasoner=self.name, tier=self.tier,
            confidence=0.5, cost_ms=self.est_cost_ms,
        )
