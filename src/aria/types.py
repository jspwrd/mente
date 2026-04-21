"""Shared types used across the system.

These correspond to concepts in the architecture:
- Event: a message on the bus (§1 Nervous System)
- Intent: a structured user/system request (§6 Interface)
- Belief: an entry in the world model (§3 World Model)
- Decision: a routing outcome (§5 Cortex)
- Trace: an auditable record of a reasoning step (§10 Verifier)
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal


def _now() -> float:
    return time.time()


def _uid(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


@dataclass
class Event:
    topic: str
    payload: dict[str, Any]
    origin: str
    trace_id: str = field(default_factory=lambda: _uid("tr_"))
    ts: float = field(default_factory=_now)
    confidence: float = 1.0


@dataclass
class Intent:
    text: str
    source: str = "user"
    trace_id: str = field(default_factory=lambda: _uid("in_"))
    ts: float = field(default_factory=_now)


@dataclass
class Belief:
    entity: str
    attribute: str
    value: Any
    confidence: float = 1.0
    ts: float = field(default_factory=_now)
    ttl_s: float | None = None  # None = permanent

    def is_live(self, now: float | None = None) -> bool:
        if self.ttl_s is None:
            return True
        return (now or _now()) - self.ts < self.ttl_s


ReasonerTier = Literal["fast", "deep", "specialist"]


@dataclass
class Decision:
    """Output of the router: which reasoner should handle this, and why."""
    tier: ReasonerTier
    reasoner: str
    reason: str
    predicted_cost_ms: float
    predicted_confidence: float


@dataclass
class Response:
    text: str
    reasoner: str
    tier: ReasonerTier
    confidence: float
    cost_ms: float
    tools_used: list[str] = field(default_factory=list)
    trace_id: str = field(default_factory=lambda: _uid("rs_"))


@dataclass
class Trace:
    """Auditable step record. Every decision, every verification, every tool call."""
    kind: str  # "decision" | "reason" | "verify" | "tool" | "state"
    actor: str
    detail: dict[str, Any]
    ts: float = field(default_factory=_now)
    trace_id: str = ""
