"""Shared types for the verifiers package.

``Verdict`` lives here (rather than in ``mente.types``) because it is the
sole data carrier between the verifier pipeline and consumers; keeping it
adjacent to the implementations avoids polluting the top-level types module.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Verdict:
    accept: bool
    score: float
    reasons: list[str] = field(default_factory=list)


__all__ = ["Verdict"]
