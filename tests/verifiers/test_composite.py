"""Tests for CompositeVerifier — merge strategies and chaining."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from aria.types import Intent, Response
from aria.verifiers import CompositeVerifier, HeuristicVerifier, Verdict
from aria.world_model import WorldModel


@dataclass
class _StubVerifier:
    """Returns a pre-canned verdict and records invocations."""

    verdict: Verdict
    calls: int = 0

    def verify(self, intent: Intent, response: Response, world: WorldModel) -> Verdict:
        self.calls += 1
        return self.verdict


# -- empty composite --------------------------------------------------------


def test_empty_composite_passes_through_confidence(intent, world, make_response_fixture):
    cv = CompositeVerifier(verifiers=[])
    out = cv.verify(intent, make_response_fixture("ok", confidence=0.9), world)
    assert out.accept is True

    out_low = cv.verify(intent, make_response_fixture("ok", confidence=0.1), world)
    assert out_low.accept is False


# -- min strategy -----------------------------------------------------------


def test_min_strategy_accepts_only_if_all_accept(intent, world, make_response_fixture):
    a = _StubVerifier(Verdict(accept=True, score=0.8, reasons=["a:ok"]))
    b = _StubVerifier(Verdict(accept=True, score=0.6, reasons=["b:ok"]))
    cv = CompositeVerifier(verifiers=[a, b], strategy="min")
    out = cv.verify(intent, make_response_fixture("hi"), world)
    assert out.accept is True
    assert out.score == pytest.approx(0.6)
    assert a.calls == 1 and b.calls == 1


def test_min_strategy_rejects_if_any_rejects(intent, world, make_response_fixture):
    a = _StubVerifier(Verdict(accept=True, score=0.9, reasons=["a:ok"]))
    b = _StubVerifier(Verdict(accept=False, score=0.2, reasons=["b:bad"]))
    cv = CompositeVerifier(verifiers=[a, b], strategy="min")
    out = cv.verify(intent, make_response_fixture("hi"), world)
    assert out.accept is False
    assert out.score == pytest.approx(0.2)


def test_min_strategy_short_circuits(intent, world, make_response_fixture):
    a = _StubVerifier(Verdict(accept=False, score=0.1, reasons=["a:bad"]))
    b = _StubVerifier(Verdict(accept=True, score=0.9, reasons=["b:ok"]))
    cv = CompositeVerifier(verifiers=[a, b], strategy="min", short_circuit=True)
    out = cv.verify(intent, make_response_fixture("hi"), world)
    assert out.accept is False
    assert a.calls == 1
    assert b.calls == 0  # short-circuited


def test_short_circuit_can_be_disabled(intent, world, make_response_fixture):
    a = _StubVerifier(Verdict(accept=False, score=0.1, reasons=["a:bad"]))
    b = _StubVerifier(Verdict(accept=True, score=0.9, reasons=["b:ok"]))
    cv = CompositeVerifier(verifiers=[a, b], strategy="min", short_circuit=False)
    cv.verify(intent, make_response_fixture("hi"), world)
    assert a.calls == 1
    assert b.calls == 1


# -- mean strategy ----------------------------------------------------------


def test_mean_strategy_averages_scores(intent, world, make_response_fixture):
    a = _StubVerifier(Verdict(accept=True, score=0.8, reasons=["a"]))
    b = _StubVerifier(Verdict(accept=True, score=0.4, reasons=["b"]))
    cv = CompositeVerifier(verifiers=[a, b], strategy="mean")
    out = cv.verify(intent, make_response_fixture("hi"), world)
    assert out.score == pytest.approx(0.6)


def test_mean_strategy_threshold_accept(intent, world, make_response_fixture):
    a = _StubVerifier(Verdict(accept=False, score=0.7, reasons=["a"]))
    b = _StubVerifier(Verdict(accept=False, score=0.7, reasons=["b"]))
    cv = CompositeVerifier(
        verifiers=[a, b], strategy="mean", accept_threshold=0.5
    )
    out = cv.verify(intent, make_response_fixture("hi"), world)
    # Children rejected but mean score clears the explicit threshold.
    assert out.accept is True


# -- any strategy -----------------------------------------------------------


def test_any_strategy_accepts_if_any_accepts(intent, world, make_response_fixture):
    a = _StubVerifier(Verdict(accept=False, score=0.2, reasons=["a:bad"]))
    b = _StubVerifier(Verdict(accept=True, score=0.9, reasons=["b:ok"]))
    cv = CompositeVerifier(verifiers=[a, b], strategy="any")
    out = cv.verify(intent, make_response_fixture("hi"), world)
    assert out.accept is True
    assert out.score == pytest.approx(0.9)


def test_any_strategy_rejects_if_all_reject(intent, world, make_response_fixture):
    a = _StubVerifier(Verdict(accept=False, score=0.2, reasons=["a"]))
    b = _StubVerifier(Verdict(accept=False, score=0.3, reasons=["b"]))
    cv = CompositeVerifier(verifiers=[a, b], strategy="any")
    out = cv.verify(intent, make_response_fixture("hi"), world)
    assert out.accept is False


# -- reasons aggregation ----------------------------------------------------


def test_reasons_are_prefixed_and_aggregated(intent, world, make_response_fixture):
    a = _StubVerifier(Verdict(accept=True, score=0.9, reasons=["a-reason"]))
    b = _StubVerifier(Verdict(accept=True, score=0.9, reasons=["b-reason"]))
    cv = CompositeVerifier(verifiers=[a, b], strategy="min")
    out = cv.verify(intent, make_response_fixture("hi"), world)
    joined = " ".join(out.reasons)
    assert "a-reason" in joined
    assert "b-reason" in joined


# -- chaining heuristic + heuristic ----------------------------------------


def test_chain_two_heuristic_verifiers(intent, world, make_response_fixture):
    strict = HeuristicVerifier(min_confidence=0.8)
    lenient = HeuristicVerifier(min_confidence=0.1)
    cv = CompositeVerifier(verifiers=[strict, lenient], strategy="min")

    # Response at mid-confidence: strict should reject, lenient should accept,
    # so under "min" the composite rejects.
    r = make_response_fixture("hello there", confidence=0.5, tier="deep")
    out = cv.verify(intent, r, world)
    assert out.accept is False

    cv_any = CompositeVerifier(verifiers=[strict, lenient], strategy="any")
    out_any = cv_any.verify(intent, r, world)
    assert out_any.accept is True


def test_unknown_strategy_raises(intent, world, make_response_fixture):
    a = _StubVerifier(Verdict(accept=True, score=0.9))
    cv = CompositeVerifier(verifiers=[a], strategy="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        cv.verify(intent, make_response_fixture("hi"), world)
