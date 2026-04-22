"""Tests for the curiosity loop.

Covers: gap detection (low accept_rate; entity with only one attribute),
tick() publishes curiosity.generate events, and the idle threshold keeps
the loop quiet when the user was active recently.
"""
from __future__ import annotations

import asyncio

import pytest

from mente.bus import EventBus
from mente.curiosity import Curiosity
from mente.state import LatentState
from mente.types import Belief, Event
from mente.world_model import WorldModel


async def _fresh(bus: EventBus | None = None) -> tuple[EventBus, WorldModel, LatentState, Curiosity]:
    b = bus or EventBus()
    if bus is None:
        await b.start()
    world = WorldModel(bus=b)
    latent = LatentState()
    curiosity = Curiosity(bus=b, world=world, latent=latent, idle_threshold_s=0.0)
    curiosity.wire()
    return b, world, latent, curiosity


@pytest.mark.asyncio
async def test_gaps_flags_low_accept_rate_and_remembered_notes() -> None:
    bus, world, latent, cur = await _fresh()
    try:
        latent.set("last_digest", {"accept_rate": 0.4, "note_count": 2, "recent_notes": ["x"]})
        gaps = cur._gaps()
        assert any("accept rate" in g for g in gaps)
        assert any("remember" in g for g in gaps)
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_gaps_flags_entity_with_only_one_attribute() -> None:
    bus, world, latent, cur = await _fresh()
    try:
        await world.assert_belief(Belief(entity="redis", attribute="uses", value="AOF"))
        gaps = cur._gaps()
        assert any("redis" in g.lower() for g in gaps)
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_gaps_do_not_fire_for_entities_with_multiple_attributes() -> None:
    bus, world, latent, cur = await _fresh()
    try:
        await world.assert_belief(Belief(entity="redis", attribute="uses", value="AOF"))
        await world.assert_belief(Belief(entity="redis", attribute="kind", value="kv-store"))
        gaps = cur._gaps()
        assert not any("redis" in g.lower() for g in gaps)
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_gaps_dedupe_across_calls() -> None:
    bus, world, latent, cur = await _fresh()
    try:
        latent.set("last_digest", {"accept_rate": 0.2, "note_count": 1})
        first = cur._gaps()
        assert first
        second = cur._gaps()
        assert second == [], "same session should not re-emit the same gaps"
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_tick_publishes_curiosity_generate_events() -> None:
    bus = EventBus()
    await bus.start()
    try:
        _, world, latent, cur = await _fresh(bus=bus)
        await world.assert_belief(Belief(entity="kafka", attribute="topic", value="events"))

        captured: list[Event] = []

        async def on_curio(e: Event) -> None:
            captured.append(e)

        bus.subscribe("curiosity.generate", on_curio, name="test.cap")

        emitted = await cur.tick()
        await asyncio.sleep(0)

        assert emitted, "expected at least one generated curiosity intent"
        assert captured, "expected at least one curiosity.generate event"
        topics = {e.topic for e in captured}
        assert topics == {"curiosity.generate"}
        assert all(e.origin == "curiosity" for e in captured)
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_tick_respects_idle_threshold_when_user_was_just_active() -> None:
    bus = EventBus()
    await bus.start()
    try:
        world = WorldModel(bus=bus)
        latent = LatentState()
        # Non-trivial idle window, so recent user activity should gate emission.
        cur = Curiosity(bus=bus, world=world, latent=latent, idle_threshold_s=10.0)
        cur.wire()
        await world.assert_belief(Belief(entity="kafka", attribute="topic", value="events"))
        await bus.publish(Event(topic="intent.user", origin="user", payload={"text": "hi"}))
        await asyncio.sleep(0)
        emitted = await cur.tick()
        assert emitted == []
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_tick_activity_tracker_ignores_internal_origins() -> None:
    """When curiosity or self re-enters the bus, it should not reset the idle
    clock — otherwise the loop would suppress itself forever."""
    bus = EventBus()
    await bus.start()
    try:
        world = WorldModel(bus=bus)
        latent = LatentState()
        cur = Curiosity(bus=bus, world=world, latent=latent, idle_threshold_s=0.5)
        cur.wire()

        # Pin the marker to a sentinel. If an internal-origin event wrongly
        # bumped it, `time.time()` would overwrite 0.0 with a positive number.
        for origin in ("curiosity", "self"):
            cur._last_user_intent_ts = 0.0
            await bus.publish(Event(topic=f"intent.{origin}", origin=origin, payload={"text": "self-prompt"}))
            await asyncio.sleep(0)
            assert cur._last_user_intent_ts == 0.0
    finally:
        await bus.close()
