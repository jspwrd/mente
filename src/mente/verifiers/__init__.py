"""Verifier package.

Exposes a ``StructuredVerifier`` Protocol plus concrete implementations.
Downstream code should import from here (``mente.verifiers``) rather than
from the legacy ``mente.verifier`` shim.
"""
from __future__ import annotations

from typing import Protocol

from ..types import Intent, Response
from ..world_model import WorldModel
from .composite import CompositeVerifier, MergeStrategy
from .heuristic import HeuristicVerifier
from .types import Verdict


class StructuredVerifier(Protocol):
    """Anything that can produce a Verdict for an (intent, response, world)."""

    def verify(self, intent: Intent, response: Response, world: WorldModel) -> Verdict:
        ...


__all__ = [
    "CompositeVerifier",
    "HeuristicVerifier",
    "MergeStrategy",
    "StructuredVerifier",
    "Verdict",
]
