"""Tests for aria.metacog.Metacog.estimate — pattern coverage + specialist bumps."""
from __future__ import annotations

from aria.metacog import Metacog
from aria.reasoners import DeepSimulatedReasoner, FastHeuristicReasoner
from aria.specialists import MathSpecialist
from aria.types import Intent


def _estimates_by_reasoner(metacog: Metacog, text: str) -> dict[str, float]:
    return {e.reasoner: e.predicted_confidence for e in metacog.estimate(Intent(text=text))}


def test_fast_covered_pattern_scores_high_on_fast():
    metacog = Metacog(reasoners=[FastHeuristicReasoner(), DeepSimulatedReasoner()])
    conf = _estimates_by_reasoner(metacog, "hello")
    assert conf["fast.heuristic"] >= 0.9


def test_fast_covered_pattern_scores_low_on_deep():
    metacog = Metacog(reasoners=[FastHeuristicReasoner(), DeepSimulatedReasoner()])
    conf = _estimates_by_reasoner(metacog, "hello")
    # Deep tier still registers, but as the general-purpose fallback (0.55).
    assert conf["deep.sim"] < conf["fast.heuristic"]


def test_unknown_pattern_fast_confidence_is_low():
    metacog = Metacog(reasoners=[FastHeuristicReasoner(), DeepSimulatedReasoner()])
    conf = _estimates_by_reasoner(metacog, "derive the riemann hypothesis")
    assert conf["fast.heuristic"] < 0.2


def test_math_specialist_bumps_on_arithmetic():
    metacog = Metacog(reasoners=[
        FastHeuristicReasoner(),
        DeepSimulatedReasoner(),
        MathSpecialist(),
    ])
    conf = _estimates_by_reasoner(metacog, "what is 2 + 2?")
    assert conf["specialist.math"] > 0.9


def test_math_specialist_dampens_deep_on_arithmetic():
    """When a specialist matches, deep confidence drops (specialist preferred)."""
    metacog = Metacog(reasoners=[
        FastHeuristicReasoner(),
        DeepSimulatedReasoner(),
        MathSpecialist(),
    ])
    conf = _estimates_by_reasoner(metacog, "compute 3 * 4")
    assert conf["deep.sim"] <= 0.35  # "specialist preferred" => 0.3


def test_deep_reasoner_gets_fallback_confidence_when_no_specialist_matches():
    metacog = Metacog(reasoners=[FastHeuristicReasoner(), DeepSimulatedReasoner()])
    conf = _estimates_by_reasoner(metacog, "explain recursion in plain english")
    # Deep is the only deep-tier option; fallback confidence is 0.55.
    assert conf["deep.sim"] == 0.55


def test_specialist_without_domain_match_gets_low_confidence():
    metacog = Metacog(reasoners=[
        FastHeuristicReasoner(),
        MathSpecialist(),
    ])
    conf = _estimates_by_reasoner(metacog, "hello there")
    # Math specialist shouldn't pick up a greeting.
    assert conf["specialist.math"] < 0.2


def test_estimate_returns_one_entry_per_reasoner():
    reasoners = [FastHeuristicReasoner(), DeepSimulatedReasoner(), MathSpecialist()]
    metacog = Metacog(reasoners=reasoners)
    ests = metacog.estimate(Intent(text="compute 1 + 1"))
    assert {e.reasoner for e in ests} == {r.name for r in reasoners}
    # Each estimate carries a rationale string.
    assert all(e.rationale for e in ests)


def test_predicted_cost_mirrors_reasoner_cost():
    reasoners = [FastHeuristicReasoner(), DeepSimulatedReasoner()]
    metacog = Metacog(reasoners=reasoners)
    ests = {e.reasoner: e.predicted_cost_ms for e in metacog.estimate(Intent(text="hi"))}
    assert ests["fast.heuristic"] == FastHeuristicReasoner().est_cost_ms
    assert ests["deep.sim"] == DeepSimulatedReasoner().est_cost_ms
