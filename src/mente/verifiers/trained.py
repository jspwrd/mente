"""Trained verifier scaffold.

``TrainedVerifier`` is a ``StructuredVerifier`` that delegates scoring to a
learned model. The model surface is deliberately minimal: anything with the
signature ``(FeatureVec) -> float`` works.

Phase 1 (this module):
    A pluggable ``Scorer`` callable plus a pure rule-based ``featurize``
    extractor. The scorer can be a constant for testing, a hand-written
    threshold rule, a logistic regression over the feature vector, or any
    other callable — no ML deps required.

Phase 2 (see ``baseline.py`` delivered by unit 5):
    Ship a concrete learned scorer — a small logistic-regression baseline
    trained on replayed traces — and wire it through the CLI (unit 6).
    Feature keys produced here MUST remain stable: the baseline model is
    trained against them and the CLI prints them.
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from ..types import Intent, Response
from ..world_model import WorldModel
from .types import Verdict

FeatureVec = dict[str, float]
Scorer = Callable[[FeatureVec], float]
"""Callable returning a probability-of-acceptance in ``[0, 1]``.

Out-of-range values are clamped by :class:`TrainedVerifier`, so scorers do
not need to guard themselves.
"""

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _rep_score(text: str) -> float:
    """Ratio of the most-common word's count to the total word count.

    Returns ``0.0`` for empty / whitespace-only text. A fully repetitive
    string (``"hi hi hi"``) scores ``1.0``; a fully unique string trends
    toward ``1 / n_tokens``.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return 0.0
    _, top = Counter(tokens).most_common(1)[0]
    return top / len(tokens)


def featurize(intent: Intent, response: Response, world: WorldModel) -> FeatureVec:
    """Extract numeric features from an (intent, response, world) triple.

    Purely rule-based — no ML deps. Keys are stable contract: downstream
    trained models and CLI inspectors index into this dict by name.

    Args:
        intent: The original ``Intent`` that produced ``response``.
        response: The candidate ``Response`` from a Reasoner.
        world: Read-only snapshot of the current world model.

    Returns:
        A ``dict[str, float]`` with the canonical feature keys.
    """
    text = response.text
    return {
        "response_len": float(len(text)),
        "response_empty": 1.0 if not text.strip() else 0.0,
        "confidence": response.confidence,
        "tool_count": float(len(response.tools_used)),
        "tier_fast": 1.0 if response.tier == "fast" else 0.0,
        "tier_specialist": 1.0 if response.tier == "specialist" else 0.0,
        "tier_deep": 1.0 if response.tier == "deep" else 0.0,
        "intent_len": float(len(intent.text)),
        "world_entity_count": float(len(list(world.entities()))),
        "repetition_score": _rep_score(text),
        "has_url": 1.0 if ("http://" in text or "https://" in text) else 0.0,
    }


@dataclass
class TrainedVerifier:
    """Verifier that defers acceptance to a learned scoring function.

    The scorer receives the output of :func:`featurize` and returns a
    probability-of-acceptance. Scores are clamped to ``[0, 1]`` and
    compared against ``threshold`` to decide ``accept``.

    Attributes:
        scorer: Any callable ``(FeatureVec) -> float``. The return value
            is clamped to ``[0, 1]``.
        threshold: Minimum (inclusive) clamped score to accept. Defaults
            to ``0.5``.
    """

    scorer: Scorer
    threshold: float = 0.5

    def verify(self, intent: Intent, response: Response, world: WorldModel) -> Verdict:
        """Score a response via the learned model and emit a ``Verdict``.

        Args:
            intent: The original ``Intent`` that produced ``response``.
            response: The candidate ``Response`` from a Reasoner.
            world: Read-only snapshot of the current world model.

        Returns:
            A ``Verdict`` whose ``score`` is clamped to ``[0, 1]`` and
            whose ``accept`` is ``score >= threshold``. ``reasons``
            always contains the stringified trained score for auditing.
        """
        features = featurize(intent, response, world)
        score = max(0.0, min(1.0, self.scorer(features)))
        accept = score >= self.threshold
        reasons = [f"trained score={score:.3f}"]
        return Verdict(accept=accept, score=score, reasons=reasons)


__all__ = ["FeatureVec", "Scorer", "TrainedVerifier", "featurize"]
