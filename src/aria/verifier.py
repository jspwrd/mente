"""Step-wise verifier.

Consulted after a reasoner produces a response. Returns accept/reject plus
a structured justification. Rejections trigger rework or escalation.

Phase 1: coarse heuristics — reject empty answers, reject responses that
contradict the world model on the same attribute, reject low-confidence
responses with no tools.

Phase 2: a trained verifier (PRM-style) plus, for formal domains, a real
checker in the loop (SMT, type system, test runner).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .types import Intent, Response
from .world_model import WorldModel


@dataclass
class Verdict:
    accept: bool
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class Verifier:
    min_confidence: float = 0.35

    def verify(self, intent: Intent, response: Response, world: WorldModel) -> Verdict:
        reasons: list[str] = []
        score = response.confidence

        if not response.text.strip():
            return Verdict(accept=False, score=0.0, reasons=["empty response"])

        if response.confidence < self.min_confidence:
            reasons.append(f"low confidence ({response.confidence:.2f} < {self.min_confidence})")
            score -= 0.2

        # Structural check: if the response claims a fact about an entity that
        # contradicts the world model, dock it.
        # (Phase 1 is coarse — real contradiction detection needs parsing.)
        for ent in world.entities():
            for attr, value in world.entity(ent).items():
                if (
                    isinstance(value, str)
                    and ent.lower() in response.text.lower()
                    and attr.lower() in response.text.lower()
                    and value.lower() not in response.text.lower()
                ):
                    reasons.append(f"possible contradiction on {ent}.{attr}")
                    score -= 0.1

        accept = score >= self.min_confidence and not any("contradiction" in r for r in reasons)
        if accept and not reasons:
            reasons.append("ok")
        return Verdict(accept=accept, score=max(0.0, min(1.0, score)), reasons=reasons)
