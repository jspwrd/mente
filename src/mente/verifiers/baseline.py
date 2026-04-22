"""Baseline trained-verifier scorer.

Trains a tiny logistic-regression model over features extracted from the
episodic memory log (``SlowMemory`` rows of ``kind="response"``) and returns a
callable ``Scorer`` consumable by ``TrainedVerifier`` (Unit 4).

Design notes:
- sklearn is pulled in lazily: fresh deployments with too little data (or no
  optional extra installed) still get a deterministic rule-based fallback.
- The fallback is stdlib-only and approximates the existing
  ``HeuristicVerifier`` signals — tier floor, confidence, tool usage — so the
  trained-verifier path is always populated with *something* reasonable.
- Training reads persisted ``response`` payloads; it does NOT require a live
  Intent/Response/WorldModel. The feature schema here is the canonical one
  that Unit 4's ``featurize`` should emit at inference time.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from ..logging import get_logger
from ..memory import SlowMemory

log = get_logger("verifiers.baseline")

# A Scorer is anything that maps a feature dict to a probability in [0, 1].
# Unit 4 consumes these; keep the alias module-local so this file can stand
# alone if the trained/ types module hasn't landed yet.
Scorer = Callable[[dict[str, float]], float]

# Canonical feature schema. Order matters: it defines the column layout fed to
# sklearn (``LogisticRegression.coef_`` lines up with this list). Unit 4's
# ``featurize`` must emit these same keys.
FEATURE_KEYS: tuple[str, ...] = (
    "confidence",        # response.confidence, [0, 1]
    "text_len",          # len(response.text), normalised via log1p
    "tool_count",        # number of tools used, normalised via log1p
    "tier_fast",         # one-hot tier indicators
    "tier_specialist",
    "tier_deep",
    "verdict_score",     # prior verdict score if present (0 otherwise)
)

# Fallback weights — chosen to mirror HeuristicVerifier ordering: high
# confidence and matching a tier-threshold both push accept; long rambling
# answers and many tools (without corroboration) nudge the other way. These
# are deliberately modest so the sigmoid output stays in a sensible range.
_FALLBACK_WEIGHTS: dict[str, float] = {
    "confidence": 3.0,
    "text_len": 0.05,
    "tool_count": -0.1,
    "tier_fast": 0.0,
    "tier_specialist": 0.2,
    "tier_deep": -0.3,
    "verdict_score": 2.0,
}
_FALLBACK_BIAS: float = -1.5


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def train_baseline(
    slow_mem: SlowMemory,
    min_samples: int = 50,
    since: float | None = None,
) -> Scorer:
    """Fit a logistic regression on ``(features, accepted?)`` pairs.

    Reads recent ``kind="response"`` episodes from ``slow_mem``, extracts a
    feature vector and a binary label (``payload["verdict"]["accept"]``) from
    each row, and fits a small sklearn ``LogisticRegression`` on the result.

    Args:
        slow_mem: Source of training data.
        min_samples: Minimum number of usable rows required before attempting
            to fit. Below this threshold we return the rule-based fallback so
            fresh deployments aren't blocked waiting for data.
        since: Optional lower bound on ``ts`` (epoch seconds). ``None`` means
            "use the full recent window".

    Returns:
        A ``Scorer`` closure: ``features_dict -> probability in [0, 1]``.

    Raises:
        ImportError: If sklearn is required (i.e. we have enough samples to
            train) but is not installed. The message points at the
            ``mente[verifier-ml]`` extra.
    """
    rows = slow_mem.query(kind="response", since=since, limit=10_000)
    samples = list(_extract_samples(rows))

    if len(samples) < min_samples:
        log.info(
            "train_baseline: %d samples < min_samples=%d, using rule-based fallback",
            len(samples),
            min_samples,
        )
        return _rule_based_scorer()

    # Degenerate-label guard: a single-class training set cannot yield a
    # meaningful logistic fit. Fall back rather than return a constant-1/0.
    labels = {label for _, label in samples}
    if len(labels) < 2:
        log.warning(
            "train_baseline: only one class in %d samples; using rule-based fallback",
            len(samples),
        )
        return _rule_based_scorer()

    LogisticRegression = _lazy_import_logreg()
    xs = [_vectorize(features) for features, _ in samples]
    ys = [int(label) for _, label in samples]

    model = LogisticRegression(solver="liblinear", max_iter=200)
    model.fit(xs, ys)
    log.info(
        "train_baseline: fit on %d samples (accept=%d reject=%d)",
        len(samples),
        sum(ys),
        len(ys) - sum(ys),
    )

    # Plain lists so the returned closure has no numpy reference — keeps
    # Unit 6's joblib persistence simple.
    coef = [float(c) for c in model.coef_[0]]
    bias = float(model.intercept_[0])

    def _score(features: dict[str, float]) -> float:
        vec = _vectorize(features)
        z = bias + sum(c * x for c, x in zip(coef, vec, strict=True))
        return _sigmoid(z)

    return _score


# ---------------------------------------------------------------------------
# fallback scorer
# ---------------------------------------------------------------------------


def _rule_based_scorer() -> Scorer:
    """Deterministic fallback: weighted linear combo of features, sigmoided.

    Used when training data is insufficient or when only one label class is
    present. Weights in ``_FALLBACK_WEIGHTS`` approximate the existing
    ``HeuristicVerifier`` signals: high confidence and specialist-tier answers
    push toward accept, deep-tier and tool-heavy answers are slightly
    penalised (they run longer and warrant extra scrutiny).
    """

    def _score(features: dict[str, float]) -> float:
        z = _FALLBACK_BIAS
        for key, weight in _FALLBACK_WEIGHTS.items():
            z += weight * float(features.get(key, 0.0))
        return _sigmoid(z)

    return _score


# ---------------------------------------------------------------------------
# feature extraction
# ---------------------------------------------------------------------------


def _extract_samples(
    rows: list[dict[str, Any]],
) -> list[tuple[dict[str, float], bool]]:
    """Pull ``(features, accept)`` pairs out of persisted response payloads.

    Rows lacking a usable verdict are skipped silently — training should be
    resilient to schema drift across versions.
    """
    out: list[tuple[dict[str, float], bool]] = []
    for row in rows:
        payload = row.get("payload")
        if not isinstance(payload, dict):
            continue
        verdict = payload.get("verdict")
        if not isinstance(verdict, dict):
            continue
        accept = verdict.get("accept")
        if not isinstance(accept, bool):
            continue
        out.append((_features_from_payload(payload), accept))
    return out


def _features_from_payload(payload: dict[str, Any]) -> dict[str, float]:
    """Rebuild the canonical feature dict from a persisted response payload.

    Mirrors Unit 4's live ``featurize`` so trained coefficients transfer. The
    one-hot tier encoding keeps the model linear while remaining readable in
    debug logs.

    The runtime (``Runtime._wire_subscribers``) persists response payloads
    with ``text``, ``tier``, ``tools``, and ``verdict`` — but NOT ``confidence``
    (that lives on the event envelope and isn't copied in). When ``confidence``
    is absent we use ``verdict.score`` and zero the ``verdict_score`` feature
    to avoid feeding the model two perfectly-collinear columns.
    """
    text = payload.get("text", "")
    tools = payload.get("tools") or []
    tier = payload.get("tier", "fast")
    verdict = payload.get("verdict") or {}
    verdict_score_raw = _as_float(verdict.get("score", 0.0))

    raw_confidence = payload.get("confidence")
    if raw_confidence is None:
        confidence = verdict_score_raw
        verdict_score = 0.0
    else:
        confidence = _as_float(raw_confidence)
        verdict_score = verdict_score_raw

    return {
        "confidence": confidence,
        "text_len": math.log1p(len(text) if isinstance(text, str) else 0),
        "tool_count": math.log1p(len(tools) if isinstance(tools, list) else 0),
        "tier_fast": 1.0 if tier == "fast" else 0.0,
        "tier_specialist": 1.0 if tier == "specialist" else 0.0,
        "tier_deep": 1.0 if tier == "deep" else 0.0,
        "verdict_score": verdict_score,
    }


def _vectorize(features: dict[str, float]) -> list[float]:
    """Project a feature dict onto the canonical column order."""
    return [float(features.get(k, 0.0)) for k in FEATURE_KEYS]


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _sigmoid(z: float) -> float:
    # Clamp to avoid overflow for absurd feature values — output stays in (0,1).
    if z >= 0:
        ez = math.exp(-min(z, 50.0))
        return 1.0 / (1.0 + ez)
    ez = math.exp(max(z, -50.0))
    return ez / (1.0 + ez)


def _lazy_import_logreg() -> Any:
    """Import ``sklearn.linear_model.LogisticRegression`` on demand.

    Raises a clean ``ImportError`` pointing at the ``mente[verifier-ml]``
    extra when sklearn is unavailable — so routine installs that never touch
    the trained verifier path don't pay the dependency cost.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is not installed; install with "
            "pip install 'mente[verifier-ml]'"
        ) from exc
    return LogisticRegression


__all__ = [
    "FEATURE_KEYS",
    "Scorer",
    "train_baseline",
]
