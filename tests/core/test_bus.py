"""Tests for aria.bus.EventBus: subscribe, publish, wildcards, fan-out, recent tap."""
from __future__ import annotations

import asyncio

import pytest

from aria.bus import EventBus
from aria.types import Event

from tests.fixtures.core_events import EventCapture


@pytest.mark.asyncio
async def test_subscribe_and_publish_delivers(bus: EventBus, event_capture: EventCapture):
    bus.subscribe("sense.audio", event_capture.handler)
    await bus.publish(Event(topic="sense.audio", payload={"v": 1}, origin="mic"))
    assert len(event_capture.events) == 1
    assert event_capture.events[0].topic == "sense.audio"
    assert event_capture.events[0].payload == {"v": 1}


@pytest.mark.asyncio
async def test_non_matching_topic_is_not_delivered(bus: EventBus, event_capture: EventCapture):
    bus.subscribe("sense.audio", event_capture.handler)
    await bus.publish(Event(topic="state.user.name", payload={}, origin="wm"))
    assert event_capture.events == []


@pytest.mark.asyncio
async def test_wildcard_pattern_matching(bus: EventBus, event_capture: EventCapture):
    bus.subscribe("sense.*", event_capture.handler)
    await bus.publish(Event(topic="sense.audio", payload={}, origin="mic"))
    await bus.publish(Event(topic="sense.vision", payload={}, origin="cam"))
    await bus.publish(Event(topic="state.user.name", payload={}, origin="wm"))
    topics = event_capture.topics()
    assert "sense.audio" in topics
    assert "sense.vision" in topics
    assert "state.user.name" not in topics


@pytest.mark.asyncio
async def test_multiple_subscribers_all_receive(bus: EventBus):
    a = EventCapture()
    b = EventCapture()
    bus.subscribe("x.*", a.handler)
    bus.subscribe("x.*", b.handler)
    await bus.publish(Event(topic="x.go", payload={}, origin="t"))
    assert len(a.events) == 1
    assert len(b.events) == 1


@pytest.mark.asyncio
async def test_fanout_runs_handlers_concurrently(bus: EventBus):
    """Handlers that each await should all progress — a strictly serial
    dispatch with a slow first handler would still work here; to assert
    concurrency we use a barrier that only completes when both handlers run."""
    started = asyncio.Event()
    proceed = asyncio.Event()
    saw_both = asyncio.Event()
    count = {"n": 0}

    async def first(_: Event) -> None:
        count["n"] += 1
        started.set()
        await proceed.wait()

    async def second(_: Event) -> None:
        # Only reachable if first didn't block the dispatcher.
        await started.wait()
        count["n"] += 1
        saw_both.set()
        proceed.set()

    bus.subscribe("t.*", first)
    bus.subscribe("t.*", second)

    await bus.publish(Event(topic="t.go", payload={}, origin="x"))
    # Both handlers must have run.
    assert saw_both.is_set()
    assert count["n"] == 2


@pytest.mark.asyncio
async def test_handler_exception_does_not_break_siblings(bus: EventBus, event_capture: EventCapture):
    async def boom(_: Event) -> None:
        raise RuntimeError("handler failed")

    bus.subscribe("t.*", boom)
    bus.subscribe("t.*", event_capture.handler)
    await bus.publish(Event(topic="t.go", payload={}, origin="x"))
    assert len(event_capture.events) == 1


@pytest.mark.asyncio
async def test_recent_returns_published_events(bus: EventBus):
    await bus.publish(Event(topic="a.1", payload={}, origin="t"))
    await bus.publish(Event(topic="a.2", payload={}, origin="t"))
    recent = bus.recent()
    topics = [e.topic for e in recent]
    assert "a.1" in topics
    assert "a.2" in topics


@pytest.mark.asyncio
async def test_recent_filters_by_pattern(bus: EventBus):
    await bus.publish(Event(topic="sense.audio", payload={}, origin="t"))
    await bus.publish(Event(topic="state.x.y", payload={}, origin="t"))
    await bus.publish(Event(topic="sense.vision", payload={}, origin="t"))
    sense_only = bus.recent("sense.*")
    assert len(sense_only) == 2
    assert all(e.topic.startswith("sense.") for e in sense_only)


@pytest.mark.asyncio
async def test_recent_respects_n_limit(bus: EventBus):
    for i in range(10):
        await bus.publish(Event(topic=f"k.{i}", payload={}, origin="t"))
    window = bus.recent(n=3)
    assert len(window) == 3
    # recent() returns the tail, so last three are k.7, k.8, k.9.
    assert [e.topic for e in window] == ["k.7", "k.8", "k.9"]


@pytest.mark.asyncio
async def test_recent_default_pattern_matches_all(bus: EventBus):
    await bus.publish(Event(topic="a.1", payload={}, origin="t"))
    await bus.publish(Event(topic="b.2", payload={}, origin="t"))
    all_recent = bus.recent()
    assert len(all_recent) == 2


@pytest.mark.asyncio
async def test_fanout_preserves_publish_order_in_tap(bus: EventBus):
    topics = ["x.1", "x.2", "x.3"]
    for t in topics:
        await bus.publish(Event(topic=t, payload={}, origin="t"))
    seen = [e.topic for e in bus.recent("x.*")]
    assert seen == topics


@pytest.mark.asyncio
async def test_close_is_safe_on_inprocess(bus: EventBus):
    # Default transport is InProcessTransport; close() should not raise.
    await bus.close()


@pytest.mark.asyncio
async def test_start_on_default_transport_is_noop(bus: EventBus, event_capture: EventCapture):
    await bus.start()
    bus.subscribe("k.*", event_capture.handler)
    await bus.publish(Event(topic="k.go", payload={}, origin="t"))
    assert len(event_capture.events) == 1
    await bus.close()
