"""Reasoner interface + stub implementations.

A Reasoner takes an Intent and the current world-model snapshot and produces
a Response. Reasoners are the swap point for real LLMs (Claude, local Llama,
specialist models).

Phase 1 ships two stubs:
  - FastHeuristicReasoner: pattern-matches simple intents instantly. Cheap,
    confident only on known shapes.
  - DeepSimulatedReasoner: simulates a heavyweight model via artificial
    latency and slightly fuzzier pattern-matching. Stand-in for a frontier LLM.

Phase 2: add AnthropicReasoner (Claude API, prompt caching), LocalReasoner
(llama.cpp), and specialist reasoners (code, math, retrieval).
"""
from __future__ import annotations

import asyncio
import random
import re
from dataclasses import dataclass
from typing import Protocol

from .tools import ToolRegistry
from .types import Intent, ReasonerTier, Response
from .world_model import WorldModel

# Forward-ref for the self-model dependency — injected at runtime to keep
# the reasoner interface clean.
_SELF_MODEL_HOOK = {"fn": None}


def set_self_model_hook(fn) -> None:  # type: ignore[no-untyped-def]
    """Allow runtime to inject a self-model query function."""
    _SELF_MODEL_HOOK["fn"] = fn


class Reasoner(Protocol):
    name: str
    tier: ReasonerTier
    est_cost_ms: float

    async def answer(
        self, intent: Intent, world: WorldModel, tools: ToolRegistry
    ) -> Response: ...


# --- Stubs ------------------------------------------------------------------

_SIMPLE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^\s*(hi|hello|hey)\b", re.I), "Hi — I'm online."),
    (re.compile(r"\bwhat time is it\b", re.I), "time:now"),  # sentinel → tool
    (re.compile(r"\bwho am i\b", re.I), "user:lookup"),
    (re.compile(r"\bremember that (.+)$", re.I), "remember:capture"),
    (re.compile(r"\bwhat do you remember\b", re.I), "remember:recall"),
    (re.compile(r"\b(what are you|what can you do|how many turns|describe yourself|your reasoners|your tools|what have you been doing)\b", re.I), "self:query"),
    (re.compile(r"\bwhat do you know about (.+)$", re.I), "semantic:search"),
]


@dataclass
class FastHeuristicReasoner:
    name: str = "fast.heuristic"
    tier: ReasonerTier = "fast"
    est_cost_ms: float = 2.0

    async def answer(
        self, intent: Intent, world: WorldModel, tools: ToolRegistry
    ) -> Response:
        tools_used: list[str] = []
        text = intent.text.strip()
        for pat, action in _SIMPLE_PATTERNS:
            m = pat.search(text)
            if not m:
                continue
            if action == "time:now":
                r = await tools.invoke("clock.now")
                tools_used.append("clock.now")
                return Response(
                    text=f"It is {r.value}.",
                    reasoner=self.name, tier=self.tier,
                    confidence=0.98, cost_ms=r.cost_ms, tools_used=tools_used,
                )
            if action == "user:lookup":
                name = world.get("user", "name")
                v = name.value if name else "someone I don't know yet"
                return Response(
                    text=f"You are {v}.",
                    reasoner=self.name, tier=self.tier,
                    confidence=0.9 if name else 0.4, cost_ms=self.est_cost_ms,
                )
            if action == "remember:capture":
                fact = m.group(1).rstrip(".!?")
                r = await tools.invoke("memory.note", fact=fact)
                tools_used.append("memory.note")
                return Response(
                    text=f"Noted: {fact}.",
                    reasoner=self.name, tier=self.tier,
                    confidence=0.95, cost_ms=r.cost_ms, tools_used=tools_used,
                )
            if action == "semantic:search":
                topic = m.group(1).rstrip(".!?")
                r = await tools.invoke("memory.search", query=topic, k=3)
                tools_used.append("memory.search")
                hits = r.value or []
                if not hits:
                    body = "nothing relevant"
                else:
                    body = "; ".join(f"{h['text']} (score {h['score']:.2f})" for h in hits)
                return Response(
                    text=f"About '{topic}': {body}.",
                    reasoner=self.name, tier=self.tier,
                    confidence=0.85 if hits else 0.4,
                    cost_ms=r.cost_ms, tools_used=tools_used,
                )
            if action == "self:query":
                hook = _SELF_MODEL_HOOK["fn"]
                reply = hook(intent.text) if hook else "self-model not attached"
                return Response(
                    text=reply,
                    reasoner=self.name, tier=self.tier,
                    confidence=0.92, cost_ms=self.est_cost_ms,
                )
            if action == "remember:recall":
                r = await tools.invoke("memory.recall")
                tools_used.append("memory.recall")
                notes = r.value or []
                body = "; ".join(notes) if notes else "nothing yet"
                return Response(
                    text=f"I remember: {body}.",
                    reasoner=self.name, tier=self.tier,
                    confidence=0.9, cost_ms=r.cost_ms, tools_used=tools_used,
                )
            return Response(
                text=action,
                reasoner=self.name, tier=self.tier,
                confidence=0.9, cost_ms=self.est_cost_ms,
            )
        # No pattern matched — low confidence, router should escalate.
        return Response(
            text="",
            reasoner=self.name, tier=self.tier,
            confidence=0.0, cost_ms=self.est_cost_ms,
        )


@dataclass
class DeepSimulatedReasoner:
    """Stand-in for a heavyweight LLM. Artificial latency; pretends to reason.

    Replace with AnthropicReasoner or LocalReasoner in Phase 2 without touching
    anything that depends on the Reasoner protocol.
    """
    name: str = "deep.sim"
    tier: ReasonerTier = "deep"
    est_cost_ms: float = 400.0

    async def answer(
        self, intent: Intent, world: WorldModel, tools: ToolRegistry
    ) -> Response:
        # Simulate a forward pass + thinking budget.
        await asyncio.sleep(self.est_cost_ms / 1000.0 * random.uniform(0.8, 1.2))
        snapshot = {e: world.entity(e) for e in world.entities()}
        # Incredibly naive "reasoning": acknowledge + echo context.
        if snapshot:
            ctx = ", ".join(f"{k}={v}" for k, v in snapshot.items())
            reply = (
                f"[deep] I thought about '{intent.text}' with context {{{ctx}}}. "
                f"I don't have a confident answer yet — wire me to a real LLM."
            )
        else:
            reply = (
                f"[deep] I considered '{intent.text}' but have no world-model context. "
                f"Wire me to a real LLM in Phase 2."
            )
        return Response(
            text=reply,
            reasoner=self.name, tier=self.tier,
            confidence=0.55, cost_ms=self.est_cost_ms,
        )
