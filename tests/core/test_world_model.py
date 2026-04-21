"""Tests for mente.world_model.WorldModel: assert_belief → state.* events + queries."""
from __future__ import annotations

import time

import pytest

from mente.bus import EventBus
from mente.types import Belief
from mente.world_model import WorldModel
from tests.fixtures.core_events import EventCapture


@pytest.mark.asyncio
async def test_assert_belief_publishes_state_event(bus: EventBus, event_capture: EventCapture):
    bus.subscribe("state.*.*", event_capture.handler)
    wm = WorldModel(bus=bus)
    await wm.assert_belief(Belief(entity="user", attribute="name", value="ada"))
    assert len(event_capture.events) == 1
    evt = event_capture.events[0]
    assert evt.topic == "state.user.name"
    assert evt.payload["entity"] == "user"
    assert evt.payload["attribute"] == "name"
    assert evt.payload["value"] == "ada"
    assert evt.payload["previous"] is None
    assert evt.origin == "world_model"


@pytest.mark.asyncio
async def test_assert_belief_propagates_confidence(bus: EventBus, event_capture: EventCapture):
    bus.subscribe("state.*.*", event_capture.handler)
    wm = WorldModel(bus=bus)
    await wm.assert_belief(Belief(entity="s", attribute="t", value=1, confidence=0.33))
    assert event_capture.events[0].confidence == 0.33


@pytest.mark.asyncio
async def test_assert_belief_emits_previous_value(bus: EventBus, event_capture: EventCapture):
    bus.subscribe("state.*.*", event_capture.handler)
    wm = WorldModel(bus=bus)
    await wm.assert_belief(Belief(entity="u", attribute="mood", value="calm"))
    await wm.assert_belief(Belief(entity="u", attribute="mood", value="excited"))
    assert len(event_capture.events) == 2
    assert event_capture.events[1].payload["previous"] == "calm"
    assert event_capture.events[1].payload["value"] == "excited"


@pytest.mark.asyncio
async def test_get_returns_current_belief(bus: EventBus):
    wm = WorldModel(bus=bus)
    await wm.assert_belief(Belief(entity="u", attribute="role", value="admin"))
    got = wm.get("u", "role")
    assert got is not None
    assert got.value == "admin"


@pytest.mark.asyncio
async def test_get_missing_returns_none(bus: EventBus):
    wm = WorldModel(bus=bus)
    assert wm.get("nobody", "anything") is None


@pytest.mark.asyncio
async def test_get_expired_belief_returns_none(bus: EventBus):
    wm = WorldModel(bus=bus)
    past = time.time() - 100.0
    await wm.assert_belief(
        Belief(entity="s", attribute="temp", value=20.0, ts=past, ttl_s=10.0)
    )
    assert wm.get("s", "temp") is None


@pytest.mark.asyncio
async def test_entity_returns_live_attributes(bus: EventBus):
    wm = WorldModel(bus=bus)
    await wm.assert_belief(Belief(entity="u", attribute="name", value="ada"))
    await wm.assert_belief(Belief(entity="u", attribute="age", value=36))
    snap = wm.entity("u")
    assert snap == {"name": "ada", "age": 36}


@pytest.mark.asyncio
async def test_entity_excludes_expired(bus: EventBus):
    wm = WorldModel(bus=bus)
    past = time.time() - 100.0
    await wm.assert_belief(Belief(entity="u", attribute="name", value="ada"))
    await wm.assert_belief(
        Belief(entity="u", attribute="session", value="old", ts=past, ttl_s=5.0)
    )
    snap = wm.entity("u")
    assert snap == {"name": "ada"}


@pytest.mark.asyncio
async def test_snapshot_returns_live_only(bus: EventBus):
    wm = WorldModel(bus=bus)
    await wm.assert_belief(Belief(entity="u", attribute="a", value=1))
    await wm.assert_belief(
        Belief(entity="x", attribute="b", value=2, ts=time.time() - 50.0, ttl_s=1.0)
    )
    snap = wm.snapshot()
    values = [(b.entity, b.attribute, b.value) for b in snap]
    assert ("u", "a", 1) in values
    assert ("x", "b", 2) not in values


@pytest.mark.asyncio
async def test_entities_lists_live_entity_names(bus: EventBus):
    wm = WorldModel(bus=bus)
    await wm.assert_belief(Belief(entity="alpha", attribute="x", value=1))
    await wm.assert_belief(Belief(entity="beta", attribute="x", value=1))
    assert set(wm.entities()) == {"alpha", "beta"}


@pytest.mark.asyncio
async def test_entities_excludes_all_expired(bus: EventBus):
    wm = WorldModel(bus=bus)
    past = time.time() - 100.0
    await wm.assert_belief(
        Belief(entity="ghost", attribute="x", value=1, ts=past, ttl_s=1.0)
    )
    assert "ghost" not in set(wm.entities())


@pytest.mark.asyncio
async def test_overwrite_replaces_stored_belief(bus: EventBus):
    wm = WorldModel(bus=bus)
    await wm.assert_belief(Belief(entity="u", attribute="score", value=1))
    await wm.assert_belief(Belief(entity="u", attribute="score", value=42))
    got = wm.get("u", "score")
    assert got is not None
    assert got.value == 42
