"""Metacognitive estimator.

Predicts — before routing — how confident each tier of reasoner is likely to
be on this intent, and how much compute it would cost. The router consumes
these predictions to decide tier.

Phase 1: hand-coded heuristics (pattern coverage = high fast-tier confidence).
Phase 2: a small trained head that takes intent embeddings + recent trace
history and outputs calibrated (confidence, cost) per reasoner.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .reasoners import Reasoner
from .types import Intent


@dataclass
class MetacogEstimate:
    reasoner: str
    predicted_confidence: float
    predicted_cost_ms: float
    rationale: str


_FAST_COVERED = [
    r"^\s*(hi|hello|hey)\b",
    r"\bwhat time is it\b",
    r"\bwho am i\b",
    r"\bremember that .+$",
    r"\bwhat do you remember\b",
    r"\b(what are you|what can you do|how many turns|describe yourself|your reasoners|your tools|what have you been doing)\b",
    r"\bwhat do you know about .+$",
]
_FAST_PATS = [re.compile(p, re.I) for p in _FAST_COVERED]

# Patterns that indicate a domain where a specialist reasoner is preferred
# over the generic deep tier. Keyed by specialization string.
_SPECIALIST_PATS: dict[str, list[re.Pattern[str]]] = {
    "math": [
        re.compile(r"\b(what is|compute|calculate|evaluate|solve)\s+[-+*/().\s0-9]+", re.I),
    ],
    "synthesis": [
        re.compile(r"\d.*\bfib(?:onacci)?\b|\bfib(?:onacci)?\b.*\d", re.I),
        re.compile(r"\bfactorial\b.*\d|\d.*\bfactorial\b", re.I),
        re.compile(r"\d+\s*(?:\*\*|\^|to the power of)\s*\d+", re.I),
    ],
}


@dataclass
class Metacog:
    reasoners: list[Reasoner] = field(default_factory=list)

    def estimate(self, intent: Intent) -> list[MetacogEstimate]:
        out: list[MetacogEstimate] = []
        fast_hit = any(p.search(intent.text) for p in _FAST_PATS)
        # Detect which specialists the intent matches.
        matched_domains = {
            dom for dom, pats in _SPECIALIST_PATS.items()
            if any(p.search(intent.text) for p in pats)
        }
        for r in self.reasoners:
            if r.tier == "fast":
                conf = 0.95 if fast_hit else 0.1
                reason = "pattern match" if fast_hit else "no pattern match"
            elif r.tier == "deep":
                # If a specialist domain matches, deep tier is a worse pick.
                conf = 0.3 if matched_domains else 0.55
                reason = "specialist preferred" if matched_domains else "general-purpose fallback"
            else:  # specialist
                # Match specialist reasoner name against domain patterns.
                name = r.name.lower()
                matched = any(dom in name for dom in matched_domains)
                conf = 0.97 if matched else 0.1
                reason = "domain match" if matched else "specialist — no domain match"
            out.append(
                MetacogEstimate(
                    reasoner=r.name,
                    predicted_confidence=conf,
                    predicted_cost_ms=r.est_cost_ms,
                    rationale=reason,
                )
            )
        return out
