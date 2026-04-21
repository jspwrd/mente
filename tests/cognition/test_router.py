"""Tests for aria.router.Router — decide() argmax + route() escalation."""
from __future__ import annotations

from aria.metacog import Metacog
from aria.router import Router
from aria.tools import ToolRegistry
from aria.types import Intent, Response

from fixtures.cognition_helpers import StubReasoner, make_world


def test_decide_picks_fast_for_fast_covered_pattern():
    reasoners = [
        StubReasoner(name="fast.heuristic", tier="fast", est_cost_ms=2.0),
        StubReasoner(name="deep.sim", tier="deep", est_cost_ms=400.0),
    ]
    router = Router(reasoners=reasoners, metacog=Metacog(reasoners=reasoners))
    decision = router.decide(Intent(text="hello"))
    assert decision.reasoner == "fast.heuristic"
    assert decision.tier == "fast"


def test_decide_picks_deep_for_unknown_pattern():
    reasoners = [
        StubReasoner(name="fast.heuristic", tier="fast", est_cost_ms=2.0),
        StubReasoner(name="deep.sim", tier="deep", est_cost_ms=400.0),
    ]
    router = Router(reasoners=reasoners, metacog=Metacog(reasoners=reasoners))
    decision = router.decide(Intent(text="derive the riemann hypothesis"))
    # Fast is 0.1, deep is 0.55 − (400/2000) = 0.35; deep wins.
    assert decision.reasoner == "deep.sim"


def test_decide_respects_ms_per_conf_tradeoff():
    """A very expensive deep reasoner should lose to a decent fast one."""
    reasoners = [
        StubReasoner(name="fast.heuristic", tier="fast", est_cost_ms=2.0),
        StubReasoner(name="deep.sim", tier="deep", est_cost_ms=1_000_000.0),
    ]
    router = Router(
        reasoners=reasoners,
        metacog=Metacog(reasoners=reasoners),
        ms_per_conf=2000.0,
    )
    decision = router.decide(Intent(text="derive the riemann hypothesis"))
    assert decision.reasoner == "fast.heuristic"


def test_decision_carries_predicted_cost_and_confidence():
    reasoners = [StubReasoner(name="fast.heuristic", tier="fast", est_cost_ms=2.0)]
    router = Router(reasoners=reasoners, metacog=Metacog(reasoners=reasoners))
    decision = router.decide(Intent(text="hello"))
    assert decision.predicted_cost_ms == 2.0
    assert decision.predicted_confidence > 0


async def test_route_escalates_when_confidence_below_threshold():
    low = Response(text="?", reasoner="fast.heuristic", tier="fast", confidence=0.1, cost_ms=2.0)
    high = Response(text="thought about it", reasoner="deep.sim", tier="deep", confidence=0.8, cost_ms=400.0)
    fast = StubReasoner(name="fast.heuristic", tier="fast", est_cost_ms=2.0, preset=low)
    deep = StubReasoner(name="deep.sim", tier="deep", est_cost_ms=400.0, preset=high)
    reasoners = [fast, deep]
    router = Router(
        reasoners=reasoners,
        metacog=Metacog(reasoners=reasoners),
        min_confidence=0.7,
    )
    world = await make_world()
    decision, response, attempted = await router.route(Intent(text="hello"), world, ToolRegistry())
    assert fast.calls == 1 and deep.calls == 1
    assert response.reasoner == "deep.sim"
    assert decision.reasoner == "deep.sim"
    assert len(attempted) == 2


async def test_route_no_escalation_when_confidence_ok():
    ok = Response(text="fine", reasoner="fast.heuristic", tier="fast", confidence=0.95, cost_ms=2.0)
    fast = StubReasoner(name="fast.heuristic", tier="fast", est_cost_ms=2.0, preset=ok)
    deep = StubReasoner(name="deep.sim", tier="deep", est_cost_ms=400.0)
    reasoners = [fast, deep]
    router = Router(
        reasoners=reasoners,
        metacog=Metacog(reasoners=reasoners),
        min_confidence=0.7,
    )
    world = await make_world()
    decision, response, attempted = await router.route(Intent(text="hello"), world, ToolRegistry())
    assert deep.calls == 0
    assert response.reasoner == "fast.heuristic"
    assert len(attempted) == 1
    assert decision.reasoner == "fast.heuristic"


async def test_route_no_escalation_when_no_deeper_tier_exists():
    low = Response(text="?", reasoner="fast.heuristic", tier="fast", confidence=0.1, cost_ms=2.0)
    fast = StubReasoner(name="fast.heuristic", tier="fast", est_cost_ms=2.0, preset=low)
    reasoners = [fast]
    router = Router(
        reasoners=reasoners,
        metacog=Metacog(reasoners=reasoners),
        min_confidence=0.7,
    )
    world = await make_world()
    decision, response, attempted = await router.route(Intent(text="hello"), world, ToolRegistry())
    assert fast.calls == 1
    assert response.reasoner == "fast.heuristic"
    assert len(attempted) == 1
    assert decision.reasoner == "fast.heuristic"


async def test_route_attempted_contains_original_and_escalation():
    low = Response(text="?", reasoner="fast.heuristic", tier="fast", confidence=0.1, cost_ms=2.0)
    high = Response(text="ok", reasoner="deep.sim", tier="deep", confidence=0.9, cost_ms=400.0)
    fast = StubReasoner(name="fast.heuristic", tier="fast", est_cost_ms=2.0, preset=low)
    deep = StubReasoner(name="deep.sim", tier="deep", est_cost_ms=400.0, preset=high)
    reasoners = [fast, deep]
    router = Router(
        reasoners=reasoners,
        metacog=Metacog(reasoners=reasoners),
        min_confidence=0.7,
    )
    world = await make_world()
    _, _, attempted = await router.route(Intent(text="hello"), world, ToolRegistry())
    assert [a.reasoner for a in attempted] == ["fast.heuristic", "deep.sim"]
    assert "escalated" in attempted[1].reason.lower()
