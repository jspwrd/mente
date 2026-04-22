"""Tests for TrainedVerifier and its companion ``featurize`` extractor."""
from __future__ import annotations

from mente.types import Belief, Intent
from mente.verifiers import (
    FeatureVec,
    TrainedVerifier,
    Verdict,
    featurize,
)

EXPECTED_FEATURE_KEYS = {
    "response_len",
    "response_empty",
    "confidence",
    "tool_count",
    "tier_fast",
    "tier_specialist",
    "tier_deep",
    "intent_len",
    "world_entity_count",
    "repetition_score",
    "has_url",
}


# -- featurize --------------------------------------------------------------


def test_featurize_typical_response_has_expected_keys(
    intent, world, make_response_fixture
):
    resp = make_response_fixture(
        "the time is 10 o'clock",
        tier="specialist",
        confidence=0.75,
        tools_used=["clock.now"],
    )
    feats = featurize(intent, resp, world)
    assert set(feats) == EXPECTED_FEATURE_KEYS
    assert all(isinstance(v, float) for v in feats.values())
    assert feats["response_len"] == float(len("the time is 10 o'clock"))
    assert feats["response_empty"] == 0.0
    assert feats["confidence"] == 0.75
    assert feats["tool_count"] == 1.0
    assert feats["tier_fast"] == 0.0
    assert feats["tier_specialist"] == 1.0
    assert feats["tier_deep"] == 0.0
    assert feats["intent_len"] == float(len(intent.text))
    assert feats["world_entity_count"] == 0.0
    assert feats["has_url"] == 0.0
    assert 0.0 <= feats["repetition_score"] <= 1.0


def test_featurize_handles_empty_text_empty_tools_empty_world(
    intent, world, make_response_fixture
):
    resp = make_response_fixture("", confidence=0.0, tools_used=[])
    feats = featurize(intent, resp, world)
    assert set(feats) == EXPECTED_FEATURE_KEYS
    assert feats["response_len"] == 0.0
    assert feats["response_empty"] == 1.0
    assert feats["tool_count"] == 0.0
    assert feats["world_entity_count"] == 0.0
    assert feats["repetition_score"] == 0.0
    assert feats["has_url"] == 0.0


def test_featurize_detects_url_and_repetition(intent, world, make_response_fixture):
    # "hi" appears 5 of 8 word-tokens ("https", "example", "com" are tokens too).
    resp = make_response_fixture("hi hi hi hi hi see https://example.com")
    feats = featurize(intent, resp, world)
    assert feats["has_url"] == 1.0
    assert feats["repetition_score"] > 0.5


def test_featurize_http_url_is_also_detected(intent, world, make_response_fixture):
    resp = make_response_fixture("visit http://neverssl.com now")
    feats = featurize(intent, resp, world)
    assert feats["has_url"] == 1.0


def test_featurize_each_tier_flag(intent, world, make_response_fixture):
    for tier, expected in [
        ("fast", "tier_fast"),
        ("deep", "tier_deep"),
        ("specialist", "tier_specialist"),
    ]:
        feats = featurize(intent, make_response_fixture("hi", tier=tier), world)
        assert feats[expected] == 1.0
        other = {"tier_fast", "tier_deep", "tier_specialist"} - {expected}
        for flag in other:
            assert feats[flag] == 0.0


def test_featurize_counts_populated_world(
    intent, world, make_response_fixture, populate_world_fixture
):
    populate_world_fixture(
        world,
        [
            Belief(entity="alice", attribute="mood", value="happy"),
            Belief(entity="bob", attribute="mood", value="sad"),
        ],
    )
    feats = featurize(intent, make_response_fixture("hello"), world)
    assert feats["world_entity_count"] == 2.0


# -- TrainedVerifier --------------------------------------------------------


def _const_scorer(value: float):
    def _scorer(_features: FeatureVec) -> float:
        return value

    return _scorer


def test_trained_verifier_accepts_with_high_score(
    intent, world, make_response_fixture
):
    v = TrainedVerifier(scorer=_const_scorer(0.9))
    out = v.verify(intent, make_response_fixture("ok"), world)
    assert isinstance(out, Verdict)
    assert out.accept is True
    assert out.score == 0.9
    assert out.reasons == ["trained score=0.900"]


def test_trained_verifier_rejects_with_low_score(
    intent, world, make_response_fixture
):
    v = TrainedVerifier(scorer=_const_scorer(0.1))
    out = v.verify(intent, make_response_fixture("ok"), world)
    assert out.accept is False
    assert out.score == 0.1


def test_threshold_is_respected(intent, world, make_response_fixture):
    # score == threshold -> accept (>=)
    boundary = TrainedVerifier(scorer=_const_scorer(0.4), threshold=0.4)
    assert boundary.verify(intent, make_response_fixture("hi"), world).accept is True

    below = TrainedVerifier(scorer=_const_scorer(0.39), threshold=0.4)
    assert below.verify(intent, make_response_fixture("hi"), world).accept is False

    strict = TrainedVerifier(scorer=_const_scorer(0.5), threshold=0.95)
    assert strict.verify(intent, make_response_fixture("hi"), world).accept is False


def test_score_is_clamped_when_scorer_returns_above_one(
    intent, world, make_response_fixture
):
    v = TrainedVerifier(scorer=_const_scorer(5.0))
    out = v.verify(intent, make_response_fixture("hi"), world)
    assert out.score == 1.0
    assert out.accept is True


def test_score_is_clamped_when_scorer_returns_negative(
    intent, world, make_response_fixture
):
    v = TrainedVerifier(scorer=_const_scorer(-2.0))
    out = v.verify(intent, make_response_fixture("hi"), world)
    assert out.score == 0.0
    assert out.accept is False


def test_scorer_receives_full_feature_vector(intent, world, make_response_fixture):
    seen: list[FeatureVec] = []

    def capture(features: FeatureVec) -> float:
        seen.append(features)
        return 0.5

    v = TrainedVerifier(scorer=capture)
    v.verify(intent, make_response_fixture("hello world"), world)
    assert len(seen) == 1
    assert set(seen[0]) == EXPECTED_FEATURE_KEYS


def test_verifier_is_a_structured_verifier_shape(intent, world, make_response_fixture):
    # Duck-typed Protocol check — just ensure .verify returns a Verdict.
    from mente.verifiers import StructuredVerifier  # noqa: F401

    v = TrainedVerifier(scorer=_const_scorer(0.6))
    out = v.verify(intent, make_response_fixture("hello"), world)
    assert isinstance(out, Verdict)
    assert isinstance(out.score, float)
    assert 0.0 <= out.score <= 1.0


def test_intent_text_feeds_intent_len(world, make_response_fixture):
    intent = Intent(text="tell me everything about the weather")
    feats = featurize(intent, make_response_fixture("it's raining"), world)
    assert feats["intent_len"] == float(len(intent.text))
