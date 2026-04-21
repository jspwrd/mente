"""Tests for HeuristicVerifier — each check gets a positive and negative case."""
from __future__ import annotations

import pytest

from aria.types import Belief, Intent
from aria.verifiers import HeuristicVerifier, Verdict


# -- basic shape / shim -----------------------------------------------------


def test_verifier_returns_verdict(intent, world, make_response_fixture):
    v = HeuristicVerifier()
    out = v.verify(intent, make_response_fixture("hello there"), world)
    assert isinstance(out, Verdict)
    assert 0.0 <= out.score <= 1.0


def test_backcompat_shim_exports_verifier_and_verdict():
    from aria.verifier import Verdict as ShimVerdict
    from aria.verifier import Verifier as ShimVerifier

    assert ShimVerifier is HeuristicVerifier
    assert ShimVerdict is Verdict


def test_min_confidence_kwarg_still_accepted():
    # Router constructs Verifier(min_confidence=...) in some paths; keep that.
    v = HeuristicVerifier(min_confidence=0.1)
    assert v.min_confidence == 0.1


# -- empty-text rejection ---------------------------------------------------


def test_empty_response_rejected(intent, world, make_response_fixture):
    v = HeuristicVerifier()
    out = v.verify(intent, make_response_fixture("   "), world)
    assert out.accept is False
    assert out.score == 0.0
    assert any("empty" in r for r in out.reasons)


def test_non_empty_response_not_trivially_rejected(intent, world, make_response_fixture):
    v = HeuristicVerifier()
    out = v.verify(intent, make_response_fixture("the time is now"), world)
    assert out.accept is True


# -- low-confidence flag ----------------------------------------------------


def test_low_confidence_flagged(intent, world, make_response_fixture):
    v = HeuristicVerifier(min_confidence=0.5)
    out = v.verify(intent, make_response_fixture("ok", confidence=0.2), world)
    assert any("low confidence" in r for r in out.reasons)
    assert out.accept is False


def test_high_confidence_not_flagged(intent, world, make_response_fixture):
    v = HeuristicVerifier(min_confidence=0.5)
    out = v.verify(intent, make_response_fixture("ok", confidence=0.9), world)
    assert not any("low confidence" in r for r in out.reasons)


# -- tier-aware thresholds --------------------------------------------------


def test_fast_tier_threshold_flags_mid_confidence(intent, world, make_response_fixture):
    v = HeuristicVerifier()
    r = make_response_fixture("fine", tier="fast", confidence=0.6)
    out = v.verify(intent, r, world)
    assert any("tier threshold" in x for x in out.reasons)


def test_deep_tier_lenient_at_mid_confidence(intent, world, make_response_fixture):
    v = HeuristicVerifier()
    r = make_response_fixture("fine", tier="deep", confidence=0.6)
    out = v.verify(intent, r, world)
    assert not any("tier threshold" in x for x in out.reasons)


def test_specialist_tier_threshold_highest(intent, world, make_response_fixture):
    v = HeuristicVerifier()
    r = make_response_fixture("fine", tier="specialist", confidence=0.75)
    out = v.verify(intent, r, world)
    assert any("tier threshold" in x for x in out.reasons)


# -- numeric range sanity ---------------------------------------------------


def test_numeric_overflow_flagged(intent, world, make_response_fixture):
    v = HeuristicVerifier()
    r = make_response_fixture("the answer = 1e200")
    out = v.verify(intent, r, world)
    assert any("overflow" in x for x in out.reasons)


def test_numeric_reasonable_not_flagged(intent, world, make_response_fixture):
    v = HeuristicVerifier()
    r = make_response_fixture("the answer = 42")
    out = v.verify(intent, r, world)
    assert not any("overflow" in x or "NaN" in x for x in out.reasons)


def test_numeric_infinity_flagged(intent, world, make_response_fixture):
    # 1e500 parses to float('inf'); the "is" branch of the regex catches it.
    v = HeuristicVerifier()
    r = make_response_fixture("x is 1e500")
    out = v.verify(intent, r, world)
    assert any("infinity" in x or "overflow" in x for x in out.reasons)


def test_numeric_equals_form_flagged(intent, world, make_response_fixture):
    v = HeuristicVerifier()
    r = make_response_fixture("the answer = 1e200")
    out = v.verify(intent, r, world)
    assert any("overflow" in x for x in out.reasons)


def test_huge_number_allowed_when_intent_asks_for_it(world, make_response_fixture):
    v = HeuristicVerifier()
    intent = Intent(text="what's a googol times a googol?")
    r = make_response_fixture("the result = 1e200")
    out = v.verify(intent, r, world)
    assert not any("overflow" in x for x in out.reasons)


# -- tool-call corroboration ------------------------------------------------


def test_memory_recall_with_no_surfaced_fact_flagged(intent, world, make_response_fixture):
    v = HeuristicVerifier()
    r = make_response_fixture("I am thinking", tools_used=["memory.recall"])
    out = v.verify(intent, r, world)
    assert any("memory.recall" in x for x in out.reasons)


def test_memory_recall_with_surfaced_fact_not_flagged(intent, world, make_response_fixture):
    v = HeuristicVerifier()
    r = make_response_fixture(
        "You previously told me the meeting is on Tuesday.",
        tools_used=["memory.recall"],
    )
    out = v.verify(intent, r, world)
    assert not any("memory.recall" in x for x in out.reasons)


def test_no_memory_recall_no_corroboration_flag(intent, world, make_response_fixture):
    v = HeuristicVerifier()
    r = make_response_fixture("short", tools_used=["clock.now"])
    out = v.verify(intent, r, world)
    assert not any("memory.recall" in x for x in out.reasons)


# -- repetition detection ---------------------------------------------------


def test_three_consecutive_tokens_flagged(intent, world, make_response_fixture):
    v = HeuristicVerifier()
    r = make_response_fixture("yes yes yes that is the answer")
    out = v.verify(intent, r, world)
    assert any("repetition" in x for x in out.reasons)


def test_no_repetition_not_flagged(intent, world, make_response_fixture):
    v = HeuristicVerifier()
    r = make_response_fixture("the answer depends on context and priorities")
    out = v.verify(intent, r, world)
    assert not any("repetition" in x for x in out.reasons)


def test_two_consecutive_tokens_not_flagged(intent, world, make_response_fixture):
    v = HeuristicVerifier()
    r = make_response_fixture("hello hello world")
    out = v.verify(intent, r, world)
    assert not any("repetition" in x for x in out.reasons)


# -- hallucinated URL -------------------------------------------------------


def test_url_in_response_flagged(intent, world, make_response_fixture):
    v = HeuristicVerifier()
    r = make_response_fixture("see https://example.com for more")
    out = v.verify(intent, r, world)
    assert any("URL" in x for x in out.reasons)


def test_no_url_not_flagged(intent, world, make_response_fixture):
    v = HeuristicVerifier()
    r = make_response_fixture("see the docs for more")
    out = v.verify(intent, r, world)
    assert not any("URL" in x for x in out.reasons)


# -- world-model contradiction ---------------------------------------------


def test_world_contradiction_flagged(intent, world, make_response_fixture, populate_world_fixture):
    populate_world_fixture(
        world,
        [Belief(entity="user", attribute="name", value="Alice")],
    )
    v = HeuristicVerifier()
    r = make_response_fixture("your user name is Bob")
    out = v.verify(intent, r, world)
    assert any("contradiction" in x for x in out.reasons)
    assert out.accept is False


def test_world_agreement_not_flagged(intent, world, make_response_fixture, populate_world_fixture):
    populate_world_fixture(
        world,
        [Belief(entity="user", attribute="name", value="Alice")],
    )
    v = HeuristicVerifier()
    r = make_response_fixture("your user name is Alice")
    out = v.verify(intent, r, world)
    assert not any("contradiction" in x for x in out.reasons)


def test_world_unmentioned_entity_not_flagged(
    intent, world, make_response_fixture, populate_world_fixture
):
    populate_world_fixture(
        world,
        [Belief(entity="user", attribute="name", value="Alice")],
    )
    v = HeuristicVerifier()
    # Response never mentions "user" → no contradiction check should fire.
    r = make_response_fixture("the weather is fine today")
    out = v.verify(intent, r, world)
    assert not any("contradiction" in x for x in out.reasons)


def test_world_non_string_value_skipped(
    intent, world, make_response_fixture, populate_world_fixture
):
    populate_world_fixture(
        world,
        [Belief(entity="user", attribute="active", value=True)],
    )
    v = HeuristicVerifier()
    r = make_response_fixture("the user active status is false")
    out = v.verify(intent, r, world)
    # bool is conservatively skipped — no contradiction flag.
    assert not any("contradiction" in x for x in out.reasons)


# -- aggregation & ok path --------------------------------------------------


def test_clean_response_reports_ok(intent, world, make_response_fixture):
    v = HeuristicVerifier()
    r = make_response_fixture("The current time is 3pm.", confidence=0.95)
    out = v.verify(intent, r, world)
    assert out.accept is True
    assert "ok" in out.reasons


def test_score_clamped_to_unit_interval(intent, world, make_response_fixture):
    v = HeuristicVerifier()
    # Pile on many negative checks to drive score below 0 without clamping.
    r = make_response_fixture(
        "yes yes yes see https://x = 1e500",
        confidence=0.1,
        tools_used=["memory.recall"],
    )
    out = v.verify(intent, r, world)
    assert 0.0 <= out.score <= 1.0


@pytest.mark.parametrize("tier", ["fast", "deep", "specialist"])
def test_all_tiers_handled(intent, world, make_response_fixture, tier):
    v = HeuristicVerifier()
    r = make_response_fixture("fine", tier=tier, confidence=0.9)
    out = v.verify(intent, r, world)
    assert isinstance(out, Verdict)
