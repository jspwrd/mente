"""Composite verifier — chain multiple ``StructuredVerifier`` implementations.

The intent is to layer a cheap heuristic check with (eventually) a trained
verifier and a domain-specific checker. The composite stays pure: it does
not know what any individual verifier does, only how to merge their verdicts.

Merge strategies:
- ``"min"`` (default): take the minimum score; accept only if all children
  accept. Most conservative.
- ``"mean"``: average score; accept if the mean is above ``accept_threshold``
  or, if unset, if every child accepts.
- ``"any"``: accept if any child accepts; score is the max.

Short-circuit: when a child rejects under the ``"min"`` strategy we can stop
evaluating further verifiers (the outcome cannot be reversed), which matters
once downstream verifiers are expensive.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from ..types import Intent, Response
from ..world_model import WorldModel
from .types import Verdict


MergeStrategy = Literal["min", "mean", "any"]


class _VerifierLike(Protocol):
    def verify(self, intent: Intent, response: Response, world: WorldModel) -> Verdict:
        ...


@dataclass
class CompositeVerifier:
    verifiers: list[_VerifierLike]
    strategy: MergeStrategy = "min"
    short_circuit: bool = True
    accept_threshold: float | None = None

    # Kept so CompositeVerifier is drop-in compatible anywhere a bare
    # ``Verifier()`` is instantiated (Router/Runtime use this arg).
    min_confidence: float = 0.35

    def verify(self, intent: Intent, response: Response, world: WorldModel) -> Verdict:
        if not self.verifiers:
            return Verdict(
                accept=response.confidence >= self.min_confidence,
                score=max(0.0, min(1.0, response.confidence)),
                reasons=["composite: no verifiers configured"],
            )

        verdicts: list[Verdict] = []
        for verifier in self.verifiers:
            v = verifier.verify(intent, response, world)
            verdicts.append(v)
            if self.short_circuit and self.strategy == "min" and not v.accept:
                break

        return self._merge(verdicts)

    def _merge(self, verdicts: list[Verdict]) -> Verdict:
        reasons = [f"[{i}] {r}" for i, v in enumerate(verdicts) for r in v.reasons]

        if self.strategy == "min":
            score = min(v.score for v in verdicts)
            accept = all(v.accept for v in verdicts)
        elif self.strategy == "mean":
            score = sum(v.score for v in verdicts) / len(verdicts)
            if self.accept_threshold is not None:
                accept = score >= self.accept_threshold
            else:
                accept = all(v.accept for v in verdicts)
        elif self.strategy == "any":
            score = max(v.score for v in verdicts)
            accept = any(v.accept for v in verdicts)
        else:
            raise ValueError(f"unknown merge strategy: {self.strategy}")

        return Verdict(accept=accept, score=max(0.0, min(1.0, score)), reasons=reasons)


__all__ = ["CompositeVerifier", "MergeStrategy"]
