"""Structured self-model.

A queryable surface describing the system's own state: which components
are loaded, recent routing behavior, latest digest, latent facts.

Phase 1: exposes structured dict via a synchronous query method.
Phase 2: becomes a first-class differentiable-memory surface consulted
during reasoning; supports counterfactual queries ("would a different
reasoner have agreed?"); is the alignment interface described in §11.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .memory import SlowMemory
from .reasoners import Reasoner
from .state import LatentState
from .tools import ToolRegistry


@dataclass
class SelfModel:
    latent: LatentState
    slow_mem: SlowMemory
    reasoners: list[Reasoner]
    tools: ToolRegistry

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
        d = self.describe()
        if "reasoner" in q or "model" in q:
            names = [r["name"] for r in d["reasoners"]]
            return f"I have {len(names)} reasoners loaded: {', '.join(names)}."
        if "tool" in q:
            names = [t["name"] for t in d["tools"]]
            return f"I have {len(names)} tools: {', '.join(names)}."
        if "turn" in q or "how many" in q:
            turns = d["latent"].get("turns", 0)
            return f"I have handled {turns} turns so far."
        if "doing" in q or "digest" in q or "summary" in q:
            digest = d["recent_digest"]
            if not digest:
                return "No consolidation has run yet."
            return (
                f"Last digest: {digest['total_responses']} responses, "
                f"accept rate {digest['accept_rate']}, "
                f"routing mix {digest['by_reasoner']}."
            )
        return "I can answer about my reasoners, tools, turns, or recent activity."
