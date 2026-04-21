"""Federation/discovery tests.

All of these run through a single in-process EventBus so remote dispatch
becomes a straightforward pub/sub check. A separate test exercises the real
TCP transport with an ephemeral port.
"""
from __future__ import annotations

import asyncio
import socket

import pytest

from mente.bus import EventBus
from mente.discovery import (
    Announcer,
    Directory,
    PeerCapability,
    RemoteReasoner,
    RemoteRequestHandler,
)
from mente.reasoners import FastHeuristicReasoner
from mente.tools import ToolRegistry
from mente.transport import TCPTransport
from mente.types import Event, Intent
from mente.world_model import WorldModel

from fixtures.advanced_helpers import EchoReasoner


@pytest.mark.asyncio
async def test_announce_once_publishes_capabilities() -> None:
    bus = EventBus()
    await bus.start()
    try:
        captured: list[Event] = []

        async def on_announce(e: Event) -> None:
            captured.append(e)

        bus.subscribe("meta.capability.announce", on_announce, name="test.capture")

        r1 = FastHeuristicReasoner()
        announcer = Announcer(bus=bus, node_id="node.A", reasoners=[r1], specialization="chat")
        await announcer.announce_once()

        assert len(captured) == 1
        p = captured[0].payload
        assert p["node_id"] == "node.A"
        assert p["reasoner"] == r1.name
        assert p["tier"] == r1.tier
        assert p["specialization"] == "chat"
        assert p["est_cost_ms"] == r1.est_cost_ms
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_directory_ingests_peers_and_filters_self() -> None:
    bus = EventBus()
    await bus.start()
    try:
        directory = Directory(bus=bus, self_node_id="node.self")
        directory.wire()

        await bus.publish(
            Event(
                topic="meta.capability.announce",
                origin="node.peer",
                payload={
                    "node_id": "node.peer",
                    "reasoner": "specialist.math",
                    "tier": "specialist",
                    "est_cost_ms": 12.0,
                    "specialization": "math",
                },
            )
        )
        # Self announcement — Directory must filter this out.
        await bus.publish(
            Event(
                topic="meta.capability.announce",
                origin="node.self",
                payload={
                    "node_id": "node.self",
                    "reasoner": "fast.heuristic",
                    "tier": "fast",
                    "est_cost_ms": 2.0,
                    "specialization": "",
                },
            )
        )
        await asyncio.sleep(0)

        specs = directory.specialists()
        assert len(specs) == 1
        assert specs[0].node_id == "node.peer"
        assert specs[0].reasoner == "specialist.math"

        # Peer map is keyed by (node_id, reasoner) and excludes self.
        assert ("node.peer", "specialist.math") in directory.peers
        assert all(k[0] != "node.self" for k in directory.peers)
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_remote_reasoner_dispatch_matching_request_id_resolves() -> None:
    """RemoteReasoner.answer publishes a remote.reason.request with a
    request_id; a matching response (same id) resolves the future."""
    bus = EventBus()
    await bus.start()
    try:
        target = PeerCapability(
            node_id="node.peer", reasoner="echo.specialist",
            tier="specialist", est_cost_ms=1.0,
        )
        remote = RemoteReasoner(bus=bus, node_id="node.self", target=target, timeout_s=5.0)

        async def on_request(e: Event) -> None:
            rid = e.payload["request_id"]
            await bus.publish(
                Event(
                    topic="remote.reason.response",
                    origin="node.peer",
                    payload={
                        "request_id": rid,
                        "text": "computed",
                        "confidence": 0.93,
                        "cost_ms": 2.5,
                    },
                )
            )

        bus.subscribe("remote.reason.request", on_request, name="test.peer.responder")

        world = WorldModel(bus=bus)
        tools = ToolRegistry()
        response = await remote.answer(Intent(text="hi there"), world, tools)
        assert response.text == "computed"
        assert response.confidence == 0.93
        assert response.reasoner.startswith("remote:node.peer:")
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_remote_reasoner_timeout_returns_fallback() -> None:
    bus = EventBus()
    await bus.start()
    try:
        target = PeerCapability(
            node_id="node.gone", reasoner="echo.specialist",
            tier="specialist", est_cost_ms=1.0,
        )
        # A very short timeout so the test runs fast.
        remote = RemoteReasoner(bus=bus, node_id="node.self", target=target, timeout_s=0.05)
        world = WorldModel(bus=bus)
        tools = ToolRegistry()

        response = await remote.answer(Intent(text="anything"), world, tools)
        assert response.confidence == 0.0
        assert "timeout" in response.text.lower()
        assert response.reasoner.startswith("remote:node.gone:")
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_remote_request_handler_dispatches_and_publishes_response() -> None:
    """RemoteRequestHandler receives remote.reason.request, delegates to a
    matching local reasoner, and publishes remote.reason.response."""
    bus = EventBus()
    await bus.start()
    try:
        reasoner = EchoReasoner()
        handler = RemoteRequestHandler(
            bus=bus, node_id="node.peer",
            reasoners=[reasoner],
            world=WorldModel(bus=bus),
            tools=ToolRegistry(),
        )
        handler.wire()

        responses: list[Event] = []

        async def on_response(e: Event) -> None:
            responses.append(e)

        bus.subscribe("remote.reason.response", on_response, name="test.resp")

        await bus.publish(
            Event(
                topic="remote.reason.request",
                origin="node.self",
                payload={
                    "request_id": "req-42",
                    "target_node": "node.peer",
                    "target_reasoner": reasoner.name,
                    "intent": "ping",
                },
            )
        )
        await asyncio.sleep(0.05)

        assert responses, "handler should publish a response for matching requests"
        r = responses[-1]
        assert r.payload["request_id"] == "req-42"
        assert r.payload["text"] == "echo:ping"
        assert r.payload["confidence"] == 0.9
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_remote_request_handler_ignores_mismatched_target_node() -> None:
    bus = EventBus()
    await bus.start()
    try:
        reasoner = EchoReasoner()
        handler = RemoteRequestHandler(
            bus=bus, node_id="node.peer",
            reasoners=[reasoner],
            world=WorldModel(bus=bus),
            tools=ToolRegistry(),
        )
        handler.wire()

        responses: list[Event] = []

        async def on_response(e: Event) -> None:
            responses.append(e)

        bus.subscribe("remote.reason.response", on_response, name="test.resp")

        await bus.publish(
            Event(
                topic="remote.reason.request",
                origin="node.self",
                payload={
                    "request_id": "req-99",
                    "target_node": "node.someone-else",
                    "target_reasoner": reasoner.name,
                    "intent": "hey",
                },
            )
        )
        await asyncio.sleep(0.02)
        assert responses == []
    finally:
        await bus.close()


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.asyncio
async def test_tcp_transport_bus_round_trip() -> None:
    """Real TCP round-trip between two EventBus instances on an ephemeral port."""
    port = _pick_free_port()
    hub_bus = EventBus(transport=TCPTransport(node_id="t.hub", port=port, role="hub"))
    spoke_bus = EventBus(transport=TCPTransport(node_id="t.spoke", port=port, role="spoke"))
    await hub_bus.start()
    await spoke_bus.start()
    try:
        inbox_hub: list[Event] = []
        inbox_spoke: list[Event] = []

        async def on_hub(e: Event) -> None:
            inbox_hub.append(e)

        async def on_spoke(e: Event) -> None:
            inbox_spoke.append(e)

        hub_bus.subscribe("t.*", on_hub, name="hub.in")
        spoke_bus.subscribe("t.*", on_spoke, name="spoke.in")

        await spoke_bus.publish(Event(topic="t.ping", payload={}, origin="t.spoke"))
        await hub_bus.publish(Event(topic="t.pong", payload={}, origin="t.hub"))
        await asyncio.sleep(0.2)

        assert any(e.origin == "t.spoke" for e in inbox_hub)
        assert any(e.origin == "t.hub" for e in inbox_spoke)
    finally:
        await spoke_bus.close()
        await hub_bus.close()


@pytest.mark.asyncio
async def test_announcer_directory_shared_bus_discovery_end_to_end() -> None:
    """Announcer + Directory on the SAME in-process bus: discovery happens
    after one announcement cycle."""
    bus = EventBus()
    await bus.start()
    try:
        directory = Directory(bus=bus, self_node_id="node.hub")
        directory.wire()

        reasoner = EchoReasoner()
        announcer = Announcer(
            bus=bus, node_id="node.peer",
            reasoners=[reasoner], specialization="math",
        )
        await announcer.announce_once()
        await asyncio.sleep(0)

        specs = directory.specialists()
        assert len(specs) == 1
        assert specs[0].node_id == "node.peer"
        assert specs[0].reasoner == reasoner.name
        assert specs[0].specialization == "math"
    finally:
        await bus.close()
