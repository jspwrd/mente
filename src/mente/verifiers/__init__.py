"""Verifier package.

Exposes a ``StructuredVerifier`` Protocol plus concrete implementations.
Downstream code should import from here (``mente.verifiers``) rather than
from the legacy ``mente.verifier`` shim.

Drop-in implementation example::

    from mente.verifiers.types import Verdict

    class AlwaysAcceptVerifier:
        def verify(self, intent, response, world):
            return Verdict(accept=True, score=1.0, reasons=["always-accept"])
"""
from __future__ import annotations

from typing import Protocol

from ..types import Intent, Response
from ..world_model import WorldModel
from .composite import CompositeVerifier, MergeStrategy
from .heuristic import HeuristicVerifier
from .trained import FeatureVec, Scorer, TrainedVerifier, featurize
from .types import Verdict


class StructuredVerifier(Protocol):
    """Anything that can produce a ``Verdict`` for an (intent, response, world).

    Verifiers run after a Reasoner returns a ``Response`` but before the
    response is surfaced to the user. mente uses them to gate low-quality
    output, trigger escalation, and attach auditable reasons to each turn.
    Multiple verifiers can be combined via ``CompositeVerifier``.

    Concurrency: ``verify`` is synchronous. mente may call verifiers
    sequentially inside a composite or in parallel across independent
    responses; implementations MUST be thread-safe and MUST NOT mutate the
    supplied ``world``. Verifiers SHOULD:

      * be fast — they run on every turn.
      * return ``Verdict(accept=False, ...)`` with concrete ``reasons``
        strings rather than raising, so the pipeline stays observable.
      * keep ``score`` in ``[0.0, 1.0]``.
    """

    def verify(self, intent: Intent, response: Response, world: WorldModel) -> Verdict:
        """Score a candidate response and decide whether to accept it.

        Args:
            intent: The original ``Intent`` that produced ``response``.
                Verifiers use it to compare what was asked with what was
                answered.
            response: The candidate ``Response`` from a Reasoner. Already
                carries its own self-assessed confidence.
            world: Read-only snapshot of the current world model. MUST
                NOT be mutated by the verifier.

        Returns:
            A ``Verdict`` with ``accept`` (bool), ``score`` in
            ``[0.0, 1.0]``, and a list of short human-readable ``reasons``
            strings recorded in the trace.

        Raises:
            Exception: Only for genuine implementation bugs. Normal
                "this response is bad" outcomes are signalled with
                ``accept=False`` and reasons, not exceptions.
        """
        ...


__all__ = [
    "CompositeVerifier",
    "FeatureVec",
    "HeuristicVerifier",
    "MergeStrategy",
    "Scorer",
    "StructuredVerifier",
    "TrainedVerifier",
    "Verdict",
    "featurize",
]
