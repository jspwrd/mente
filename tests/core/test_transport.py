"""Tests for aria.transport: InProcessTransport no-op + TCP hub/spoke flow."""
from __future__ import annotations

import asyncio

import pytest

from aria.transport import InProcessTransport, TCPTransport
from aria.types import Event

from tests.fixtures.core_events import EventCapture
from tests.fixtures.core_net import find_unused_port


async def _noop(_e: Event) -> None:
    return None


@pytest.mark.asyncio
async def test_inprocess_transport_start_and_close():
    t = InProcessTransport()
    cap = EventCapture()
    await t.start(cap.handler)
    await t.publish_remote(Event(topic="x", payload={}, origin="self"))
    await t.close()
    # In-process transport never delivers anything remotely.
    assert cap.events == []


@pytest.mark.asyncio
async def test_inprocess_publish_remote_is_noop():
    t = InProcessTransport()
    # No start, no handler. Must not raise.
    await t.publish_remote(Event(topic="x", payload={}, origin="self"))


@pytest.mark.asyncio
async def test_tcp_hub_and_spoke_roundtrip():
    port = find_unused_port()
    hub = TCPTransport(node_id="hub", port=port, role="hub")
    spoke = TCPTransport(node_id="spoke", port=port, role="spoke")

    hub_rx = EventCapture()
    spoke_rx = EventCapture()

    await hub.start(hub_rx.handler)
    try:
        await spoke.start(spoke_rx.handler)
        try:
            await spoke.publish_remote(Event(topic="s.1", payload={"v": 1}, origin="spoke"))
            await asyncio.sleep(0.1)
            assert "s.1" in hub_rx.topics()

            await hub.publish_remote(Event(topic="h.1", payload={"v": 2}, origin="hub"))
            await asyncio.sleep(0.1)
            assert "h.1" in spoke_rx.topics()
        finally:
            await spoke.close()
    finally:
        await hub.close()


@pytest.mark.asyncio
async def test_tcp_spoke_ignores_self_echo():
    """When the hub rebroadcasts a spoke's event, the spoke must drop it
    because origin matches its node_id."""
    port = find_unused_port()
    hub = TCPTransport(node_id="hub", port=port, role="hub")
    spoke_a = TCPTransport(node_id="a", port=port, role="spoke")
    spoke_b = TCPTransport(node_id="b", port=port, role="spoke")

    hub_rx = EventCapture()
    a_rx = EventCapture()
    b_rx = EventCapture()

    await hub.start(hub_rx.handler)
    try:
        await spoke_a.start(a_rx.handler)
        await spoke_b.start(b_rx.handler)
        await asyncio.sleep(0.05)  # let hub accept both spokes
        try:
            await spoke_a.publish_remote(Event(topic="fan.1", payload={}, origin="a"))
            await asyncio.sleep(0.15)

            assert "fan.1" in hub_rx.topics()
            assert "fan.1" in b_rx.topics()
            assert "fan.1" not in a_rx.topics()
        finally:
            await spoke_a.close()
            await spoke_b.close()
    finally:
        await hub.close()


@pytest.mark.asyncio
async def test_tcp_multiple_spokes_all_receive_hub_broadcast():
    port = find_unused_port()
    hub = TCPTransport(node_id="hub", port=port, role="hub")
    spoke_a = TCPTransport(node_id="a", port=port, role="spoke")
    spoke_b = TCPTransport(node_id="b", port=port, role="spoke")

    a_rx = EventCapture()
    b_rx = EventCapture()

    await hub.start(_noop)
    try:
        await spoke_a.start(a_rx.handler)
        await spoke_b.start(b_rx.handler)
        await asyncio.sleep(0.05)  # let hub register both writers
        try:
            await hub.publish_remote(Event(topic="bc.1", payload={}, origin="hub"))
            await asyncio.sleep(0.15)
            assert "bc.1" in a_rx.topics()
            assert "bc.1" in b_rx.topics()
        finally:
            await spoke_a.close()
            await spoke_b.close()
    finally:
        await hub.close()


@pytest.mark.asyncio
async def test_tcp_close_is_graceful_when_unstarted():
    t = TCPTransport(node_id="n", port=find_unused_port(), role="hub")
    await t.close()


@pytest.mark.asyncio
async def test_tcp_close_is_graceful_after_start():
    port = find_unused_port()
    hub = TCPTransport(node_id="hub", port=port, role="hub")
    spoke = TCPTransport(node_id="s", port=port, role="spoke")

    await hub.start(_noop)
    await spoke.start(_noop)
    await spoke.close()
    await hub.close()


@pytest.mark.asyncio
async def test_tcp_spoke_publish_without_connection_is_noop():
    """If a spoke is unstarted, publish_remote must silently drop."""
    t = TCPTransport(node_id="s", port=find_unused_port(), role="spoke")
    await t.publish_remote(Event(topic="x", payload={}, origin="s"))
    await t.close()


@pytest.mark.asyncio
async def test_tcp_event_fields_preserved_across_wire():
    port = find_unused_port()
    hub = TCPTransport(node_id="hub", port=port, role="hub")
    spoke = TCPTransport(node_id="spoke-xyz", port=port, role="spoke")

    hub_rx = EventCapture()
    await hub.start(hub_rx.handler)
    try:
        await spoke.start(_noop)
        try:
            await spoke.publish_remote(
                Event(
                    topic="chk",
                    payload={"k": "v", "n": 3},
                    origin="spoke-xyz",
                    confidence=0.42,
                )
            )
            await asyncio.sleep(0.1)
            got = next((e for e in hub_rx.events if e.topic == "chk"), None)
            assert got is not None, "hub did not receive event"
            assert got.origin == "spoke-xyz"
            assert got.payload == {"k": "v", "n": 3}
            assert got.confidence == 0.42
        finally:
            await spoke.close()
    finally:
        await hub.close()


@pytest.mark.asyncio
async def test_tcp_hub_rebroadcast_excludes_sender():
    """Two spokes; one sends; only the other should receive (sender is excluded)."""
    port = find_unused_port()
    hub = TCPTransport(node_id="hub", port=port, role="hub")
    sender = TCPTransport(node_id="sender", port=port, role="spoke")
    receiver = TCPTransport(node_id="receiver", port=port, role="spoke")

    sender_rx = EventCapture()
    receiver_rx = EventCapture()

    await hub.start(_noop)
    try:
        await sender.start(sender_rx.handler)
        await receiver.start(receiver_rx.handler)
        await asyncio.sleep(0.05)  # let hub register both writers
        try:
            await sender.publish_remote(Event(topic="one", payload={}, origin="sender"))
            await asyncio.sleep(0.15)
            assert "one" in receiver_rx.topics()
            assert "one" not in sender_rx.topics()
        finally:
            await sender.close()
            await receiver.close()
    finally:
        await hub.close()
