"""Structured self-model.

A queryable surface describing the system's own state: which components
are loaded, recent routing behavior, latest digest, latent facts.

Phase 1: exposes structured dict via a synchronous query method.
Phase 2: becomes a first-class differentiable-memory surface consulted
during reasoning; supports counterfactual queries ("would a different
reasoner have agreed?"); is the alignment interface described in §11.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .memory import SlowMemory
from .reasoners import Reasoner
from .state import LatentState
from .tools import ToolRegistry

# A single dispatch entry: a tuple of substring keywords that match the
# lowercased question, paired with a handler that turns the describe()
# snapshot into a user-facing reply. Order defines precedence.
DispatchHandler = Callable[[dict[str, Any]], str]
DispatchEntry = tuple[tuple[str, ...], DispatchHandler]


@dataclass
class SelfModel:
    latent: LatentState
    slow_mem: SlowMemory
    reasoners: list[Reasoner]
    tools: ToolRegistry
    _dispatch: list[DispatchEntry] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Precedence: reasoner/model > tool > turn/count > doing/digest/summary.
        # If no entry matches the question, answer() falls through to _fallback.
        self._dispatch = [
            (("reasoner", "model"), self._describe_reasoners),
            (("tool",), self._describe_tools),
            (("turn", "how many"), self._describe_turns),
            (("doing", "digest", "summary"), self._describe_digest),
        ]

    def describe(self) -> dict[str, Any]:
        return {
            "reasoners": [
                {"name": r.name, "tier": r.tier, "est_cost_ms": r.est_cost_ms}
                for r in self.reasoners
            ],
            "tools": [
                {"name": t.name, "description": t.description, "returns": t.returns}
                for t in self.tools.list()
            ],
            "latent": dict(self.latent.values),
            "recent_digest": self.latent.get("last_digest"),
        }

    def answer(self, question: str) -> str:
        q = question.lower()
        desc = self.describe()
        for keywords, handler in self._dispatch:
            if any(k in q for k in keywords):
                return handler(desc)
        return self._fallback(desc)

    # -- handlers -----------------------------------------------------------

    def _describe_reasoners(self, desc: dict[str, Any]) -> str:
        names = [r["name"] for r in desc["reasoners"]]
        return f"I have {len(names)} reasoners loaded: {', '.join(names)}."

    def _describe_tools(self, desc: dict[str, Any]) -> str:
        names = [t["name"] for t in desc["tools"]]
        return f"I have {len(names)} tools: {', '.join(names)}."

    def _describe_turns(self, desc: dict[str, Any]) -> str:
        turns = desc["latent"].get("turns", 0)
        return f"I have handled {turns} turns so far."

    def _describe_digest(self, desc: dict[str, Any]) -> str:
        digest = desc["recent_digest"]
        if not digest:
            return "No consolidation has run yet."
        return (
            f"Last digest: {digest['total_responses']} responses, "
            f"accept rate {digest['accept_rate']}, "
            f"routing mix {digest['by_reasoner']}."
        )

    def _fallback(self, desc: dict[str, Any]) -> str:
        return "I can answer about my reasoners, tools, turns, or recent activity."
