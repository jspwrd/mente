"""Tests for aria.types dataclass defaults and TTL semantics."""
from __future__ import annotations

import time

from aria.types import Belief, Decision, Event, Intent, Response, Trace


def test_event_defaults_populated():
    e = Event(topic="sense.audio", payload={"x": 1}, origin="mic")
    assert e.topic == "sense.audio"
    assert e.payload == {"x": 1}
    assert e.origin == "mic"
    assert e.confidence == 1.0
    assert isinstance(e.ts, float)
    assert e.ts > 0
    assert e.trace_id.startswith("tr_")
    # 3-char prefix + 12 hex chars.
    assert len(e.trace_id) == 15


def test_event_trace_ids_are_unique():
    a = Event(topic="a", payload={}, origin="x")
    b = Event(topic="a", payload={}, origin="x")
    assert a.trace_id != b.trace_id


def test_intent_defaults():
    i = Intent(text="hello")
    assert i.text == "hello"
    assert i.source == "user"
    assert i.trace_id.startswith("in_")
    assert len(i.trace_id) == 15
    assert isinstance(i.ts, float)


def test_response_defaults():
    r = Response(text="ok", reasoner="fast_llm", tier="fast", confidence=0.9, cost_ms=12.5)
    assert r.tools_used == []
    assert r.trace_id.startswith("rs_")
    assert len(r.trace_id) == 15


def test_decision_shape():
    d = Decision(
        tier="deep",
        reasoner="planner",
        reason="multi-step task",
        predicted_cost_ms=150.0,
        predicted_confidence=0.7,
    )
    assert d.tier == "deep"
    assert d.reasoner == "planner"
    assert d.predicted_cost_ms == 150.0


def test_trace_defaults():
    t = Trace(kind="tool", actor="registry", detail={"tool": "calc"})
    # Trace.trace_id defaults to empty string (caller fills it in).
    assert t.trace_id == ""
    assert isinstance(t.ts, float)
    assert t.kind == "tool"
    assert t.actor == "registry"
    assert t.detail == {"tool": "calc"}


def test_belief_permanent_is_live():
    b = Belief(entity="user", attribute="name", value="ada")
    assert b.ttl_s is None
    assert b.is_live() is True


def test_belief_ttl_expires():
    # ts in the past, ttl small → not live.
    past = time.time() - 100.0
    b = Belief(entity="sensor", attribute="temp", value=21.5, ts=past, ttl_s=10.0)
    assert b.is_live() is False


def test_belief_ttl_still_live():
    b = Belief(entity="sensor", attribute="temp", value=21.5, ttl_s=60.0)
    assert b.is_live() is True


def test_belief_is_live_with_explicit_now():
    b = Belief(entity="x", attribute="y", value=1, ts=1000.0, ttl_s=50.0)
    assert b.is_live(now=1040.0) is True
    assert b.is_live(now=1060.0) is False


def test_belief_confidence_default():
    b = Belief(entity="x", attribute="y", value=1)
    assert b.confidence == 1.0
