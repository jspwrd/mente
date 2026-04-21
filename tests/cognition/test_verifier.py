"""Tests for mente.verifier.Verifier — accept/reject + contradiction detection."""
from __future__ import annotations

from fixtures.cognition_helpers import make_world

from mente.types import Belief, Intent, Response
from mente.verifier import Verifier


def _resp(text: str, conf: float = 0.9) -> Response:
    return Response(text=text, reasoner="x", tier="fast", confidence=conf, cost_ms=1.0)


async def test_reject_empty_response():
    world = await make_world()
    v = Verifier()
    verdict = v.verify(Intent(text="hi"), _resp("", conf=0.95), world)
    assert verdict.accept is False
    assert verdict.score == 0.0
    assert any("empty" in r for r in verdict.reasons)


async def test_reject_whitespace_only_response():
    world = await make_world()
    v = Verifier()
    verdict = v.verify(Intent(text="hi"), _resp("   \n\t", conf=0.95), world)
    assert verdict.accept is False


async def test_accept_normal_response():
    world = await make_world()
    v = Verifier()
    verdict = v.verify(Intent(text="hi"), _resp("hello there"), world)
    assert verdict.accept is True
    assert verdict.score > 0.5


async def test_flag_low_confidence_reason_added():
    """Below the minimum: reason string recorded and verdict rejected."""
    world = await make_world()
    v = Verifier(min_confidence=0.5)
    verdict = v.verify(Intent(text="hi"), _resp("something", conf=0.3), world)
    assert any("low confidence" in r for r in verdict.reasons)
    assert verdict.accept is False


async def test_reject_very_low_confidence():
    world = await make_world()
    v = Verifier(min_confidence=0.35)
    verdict = v.verify(Intent(text="hi"), _resp("text", conf=0.1), world)
    assert verdict.accept is False
    assert any("low confidence" in r for r in verdict.reasons)


async def test_accept_above_threshold_with_high_confidence():
    world = await make_world()
    v = Verifier(min_confidence=0.35)
    verdict = v.verify(Intent(text="hi"), _resp("response text", conf=0.9), world)
    assert verdict.accept is True
    assert "ok" in verdict.reasons


async def test_contradiction_on_entity_attribute_flagged():
    world = await make_world([Belief(entity="user", attribute="name", value="Ada")])
    v = Verifier()
    # Response mentions user.name but gives a different value.
    verdict = v.verify(
        Intent(text="who am I"),
        _resp("Your user name is Bob.", conf=0.9),
        world,
    )
    assert any("contradiction" in r for r in verdict.reasons)
    assert verdict.accept is False


async def test_no_contradiction_when_value_present():
    world = await make_world([Belief(entity="user", attribute="name", value="Ada")])
    v = Verifier()
    verdict = v.verify(
        Intent(text="who am I"),
        _resp("Your user name is Ada.", conf=0.9),
        world,
    )
    assert not any("contradiction" in r for r in verdict.reasons)
    assert verdict.accept is True


async def test_no_contradiction_when_entity_unmentioned():
    """Contradiction check only fires when the response actually mentions the entity."""
    world = await make_world([Belief(entity="user", attribute="name", value="Ada")])
    v = Verifier()
    verdict = v.verify(Intent(text="hi"), _resp("greetings traveller", conf=0.9), world)
    assert verdict.accept is True
    assert not any("contradiction" in r for r in verdict.reasons)


async def test_score_clamped_to_unit_interval():
    world = await make_world()
    v = Verifier()
    verdict = v.verify(Intent(text="hi"), _resp("fine", conf=1.5), world)
    assert 0.0 <= verdict.score <= 1.0
