"""Richer heuristic verifier.

Layered checks, each returning a (score_delta, reason) pair so the final
verdict aggregates transparently. Verifier is consulted after a reasoner
produces a response and gates acceptance/escalation.

Design notes:
- Score starts at ``response.confidence`` and is docked by failing checks.
- Tier-aware thresholds: fast reasoners are held to higher bars than deep
  ones (deep tiers are expected to think longer and produce lower-confidence
  exploratory answers that are still useful).
- World-model contradiction is intentionally conservative — this is a phase-1
  heuristic and false positives are worse than missed contradictions (they
  would trigger needless rework).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from ..types import Intent, Response
from ..world_model import WorldModel
from .types import Verdict


# Tier-aware acceptance thresholds. See module docstring for rationale.
TIER_THRESHOLDS: dict[str, float] = {
    "fast": 0.7,
    "specialist": 0.8,
    "deep": 0.5,
}


# A number that follows "=" or " is " (with optional whitespace). We keep the
# regex simple — the check is a sanity ward, not a parser.
_NUM_AFTER_EQ_OR_IS = re.compile(
    r"(?:=|\bis\b)\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
)
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Ceiling above which a response-asserted number is treated as nonsense
# unless the intent explicitly requested something that big.
_HUGE = 1e100

# Phrases in the intent that legitimately invite astronomically large numbers.
_BIG_NUMBER_CUES = (
    "googol",
    "factorial",
    "exponent",
    "astronom",
    "infinit",
    "huge number",
    "large number",
)


CheckResult = tuple[float, str | None]


@dataclass
class HeuristicVerifier:
    """Verifier that composes several cheap, local heuristics.

    ``min_confidence`` is kept as a constructor arg for back-compat with the
    old ``Verifier`` class — ``Router``/``Runtime`` instantiate this as
    ``Verifier()`` via the re-export shim.
    """

    min_confidence: float = 0.35
    tier_thresholds: dict[str, float] = field(
        default_factory=lambda: dict(TIER_THRESHOLDS)
    )

    # -- public --------------------------------------------------------------
    def verify(self, intent: Intent, response: Response, world: WorldModel) -> Verdict:
        if not response.text.strip():
            return Verdict(accept=False, score=0.0, reasons=["empty response"])

        reasons: list[str] = []
        score = response.confidence
        hard_fail = False

        for delta, reason in self._run_checks(intent, response, world):
            if reason is not None:
                reasons.append(reason)
                if reason.startswith("contradiction"):
                    hard_fail = True
            score += delta

        accept = (not hard_fail) and score >= self.min_confidence
        if accept and not reasons:
            reasons.append("ok")
        return Verdict(accept=accept, score=max(0.0, min(1.0, score)), reasons=reasons)

    # -- check pipeline ------------------------------------------------------
    def _run_checks(
        self, intent: Intent, response: Response, world: WorldModel
    ) -> list[CheckResult]:
        return [
            self._check_low_confidence(response),
            self._check_tier_threshold(response),
            self._check_numeric_sanity(intent, response),
            self._check_tool_corroboration(response),
            self._check_repetition(response),
            self._check_hallucinated_url(response),
            self._check_world_contradiction(response, world),
        ]

    # -- individual checks ---------------------------------------------------
    def _check_low_confidence(self, response: Response) -> CheckResult:
        if response.confidence < self.min_confidence:
            return (
                -0.2,
                f"low confidence ({response.confidence:.2f} < {self.min_confidence})",
            )
        return (0.0, None)

    def _check_tier_threshold(self, response: Response) -> CheckResult:
        threshold = self.tier_thresholds.get(response.tier)
        if threshold is None:
            return (0.0, None)
        if response.confidence < threshold:
            return (
                -0.05,
                (
                    f"below tier threshold "
                    f"({response.tier}: {response.confidence:.2f} < {threshold:.2f})"
                ),
            )
        return (0.0, None)

    def _check_numeric_sanity(self, intent: Intent, response: Response) -> CheckResult:
        matches = _NUM_AFTER_EQ_OR_IS.findall(response.text)
        if not matches:
            return (0.0, None)
        intent_lower = intent.text.lower()
        expects_big = any(cue in intent_lower for cue in _BIG_NUMBER_CUES)
        for raw in matches:
            try:
                value = float(raw)
            except ValueError:
                continue
            if math.isnan(value):
                return (-0.3, f"numeric NaN in response ({raw})")
            if expects_big:
                continue
            if math.isinf(value):
                return (-0.2, f"numeric infinity in response ({raw})")
            if abs(value) > _HUGE:
                return (-0.2, f"numeric overflow in response ({raw})")
        return (0.0, None)

    def _check_tool_corroboration(self, response: Response) -> CheckResult:
        if "memory.recall" not in response.tools_used:
            return (0.0, None)
        text = response.text.lower()
        noted_markers = ("note", "recall", "remember", "you told me", "earlier", "previously")
        has_marker = any(m in text for m in noted_markers)
        has_structure = any(ch in response.text for ch in ("\"", "'", "- ", "• ", "\n"))
        if not has_marker and not has_structure:
            return (
                -0.1,
                "memory.recall used but no noted fact surfaced in response",
            )
        return (0.0, None)

    def _check_repetition(self, response: Response) -> CheckResult:
        tokens = _TOKEN_RE.findall(response.text.lower())
        if len(tokens) < 3:
            return (0.0, None)
        run = 1
        for i in range(1, len(tokens)):
            if tokens[i] == tokens[i - 1]:
                run += 1
                if run >= 3:
                    return (
                        -0.15,
                        f"repetition detected ({tokens[i]!r} x{run})",
                    )
            else:
                run = 1
        return (0.0, None)

    def _check_hallucinated_url(self, response: Response) -> CheckResult:
        if _URL_RE.search(response.text):
            return (-0.2, "hallucinated URL in response")
        return (0.0, None)

    def _check_world_contradiction(
        self, response: Response, world: WorldModel
    ) -> CheckResult:
        """Conservative contradiction check.

        Only flags when *all* of the following are true:
        - the response mentions both an entity AND one of its attributes
        - the world has a string/number value for that attribute
        - the response does NOT contain that value

        We skip attributes whose stored value is structured (list/dict/bool)
        since naive text containment can't reason about those.
        """
        text_lower = response.text.lower()
        for ent in world.entities():
            if ent.lower() not in text_lower:
                continue
            for attr, value in world.entity(ent).items():
                if not isinstance(value, (str, int, float)):
                    continue
                if isinstance(value, bool):
                    # bool is a subclass of int — skip it: can't detect
                    # "true"/"false" contradictions without semantic parsing.
                    continue
                if attr.lower() not in text_lower:
                    continue
                value_str = str(value).lower()
                if not value_str:
                    continue
                if value_str in text_lower:
                    continue
                return (
                    -0.1,
                    f"contradiction on {ent}.{attr}",
                )
        return (0.0, None)


__all__ = ["HeuristicVerifier", "TIER_THRESHOLDS"]
