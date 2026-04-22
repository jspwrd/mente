"""Federation/discovery tests.

All of these run through a single in-process EventBus so remote dispatch
becomes a straightforward pub/sub check. A separate test exercises the real
TCP transport with an ephemeral port.
"""
from __future__ import annotations

import asyncio
import socket

import pytest
from fixtures.advanced_helpers import EchoReasoner

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
async def test_remote_reasoner_timeout_clears_pending() -> None:
    """After a timeout, the request must not remain in `_pending` — the
    finally block is the backstop against long-running leaks."""
    bus = EventBus()
    await bus.start()
    try:
        target = PeerCapability(
            node_id="node.gone", reasoner="echo.specialist",
            tier="specialist", est_cost_ms=1.0,
        )
        remote = RemoteReasoner(bus=bus, node_id="node.self", target=target, timeout_s=0.02)
        world = WorldModel(bus=bus)
        tools = ToolRegistry()

        await remote.answer(Intent(text="first"), world, tools)
        await remote.answer(Intent(text="second"), world, tools)
        assert remote._pending == {}
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_remote_reasoner_resolves_clears_pending() -> None:
    """Successful resolution must also clean up `_pending`."""
    bus = EventBus()
    await bus.start()
    try:
        target = PeerCapability(
            node_id="node.peer", reasoner="echo.specialist",
            tier="specialist", est_cost_ms=1.0,
        )
        remote = RemoteReasoner(bus=bus, node_id="node.self", target=target, timeout_s=5.0)

        async def on_request(e: Event) -> None:
            await bus.publish(
                Event(
                    topic="remote.reason.response",
                    origin="node.peer",
                    payload={
                        "request_id": e.payload["request_id"],
                        "text": "ok",
                        "confidence": 0.8,
                        "cost_ms": 1.0,
                    },
                )
            )

        bus.subscribe("remote.reason.request", on_request, name="test.peer.responder")
        await remote.answer(Intent(text="hi"), world=WorldModel(bus=bus), tools=ToolRegistry())
        assert remote._pending == {}
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_remote_reasoner_prune_stale_drops_abandoned_futures() -> None:
    """Synthetic stale entry older than 2*timeout_s is pruned on next answer."""
    bus = EventBus()
    await bus.start()
    try:
        target = PeerCapability(
            node_id="node.peer", reasoner="echo.specialist",
            tier="specialist", est_cost_ms=1.0,
        )
        remote = RemoteReasoner(bus=bus, node_id="node.self", target=target, timeout_s=0.02)

        from mente.discovery import _PendingFuture
        stale_fut: asyncio.Future[dict[str, object]] = asyncio.get_event_loop().create_future()
        remote._wire()
        remote._pending["stale-id"] = _PendingFuture(
            future=stale_fut,
            created_at=0.0,  # ancient: monotonic clock starts near 0 but grows; cutoff is now - 2*timeout
        )
        assert "stale-id" in remote._pending

        # Run an answer — it will time out, and the prune step (on entry)
        # should evict the synthetic stale entry first.
        await remote.answer(Intent(text="trigger"), WorldModel(bus=bus), ToolRegistry())
        assert "stale-id" not in remote._pending
        assert stale_fut.cancelled()
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
async def test_directory_evicts_peer_after_stale_after_s() -> None:
    """With no fresh announcement, ``specialists()`` drops peers older than
    ``stale_after_s`` on the next call."""
    bus = EventBus()
    await bus.start()
    try:
        directory = Directory(bus=bus, self_node_id="node.self", stale_after_s=0.05)
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
        await asyncio.sleep(0)
        assert len(directory.specialists()) == 1

        # Wait past the staleness window — the next specialists() call must sweep.
        await asyncio.sleep(0.1)
        assert directory.specialists() == []
        assert ("node.peer", "specialist.math") not in directory.peers
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_directory_fresh_announcement_resets_last_seen() -> None:
    """A second announcement after half the window bumps ``last_seen`` so the
    peer survives the next sweep."""
    bus = EventBus()
    await bus.start()
    try:
        directory = Directory(bus=bus, self_node_id="node.self", stale_after_s=0.1)
        directory.wire()

        payload = {
            "node_id": "node.peer",
            "reasoner": "specialist.math",
            "tier": "specialist",
            "est_cost_ms": 12.0,
            "specialization": "math",
        }
        await bus.publish(
            Event(topic="meta.capability.announce", origin="node.peer", payload=payload)
        )
        await asyncio.sleep(0)
        first_seen = directory.peers[("node.peer", "specialist.math")].last_seen

        # Re-announce before expiry — refresh resets last_seen.
        await asyncio.sleep(0.06)
        await bus.publish(
            Event(topic="meta.capability.announce", origin="node.peer", payload=payload)
        )
        await asyncio.sleep(0)

        refreshed = directory.peers[("node.peer", "specialist.math")].last_seen
        assert refreshed > first_seen
        # Still present — the refresh saved it from the staleness cutoff.
        assert len(directory.specialists()) == 1
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_directory_run_sweeps_proactively() -> None:
    """The ``run`` background task evicts stale peers even if nobody calls
    ``specialists()`` — mirrors the Announcer heartbeat."""
    bus = EventBus()
    await bus.start()
    try:
        directory = Directory(bus=bus, self_node_id="node.self", stale_after_s=0.05)
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
        await asyncio.sleep(0)
        assert ("node.peer", "specialist.math") in directory.peers

        stop = asyncio.Event()
        task = asyncio.create_task(directory.run(stop, interval_s=0.02))
        try:
            # Allow the sweep to fire after the staleness window passes.
            await asyncio.sleep(0.15)
            assert directory.peers == {}
        finally:
            stop.set()
            await task
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_remote_reasoner_opens_breaker_after_consecutive_timeouts() -> None:
    """Three consecutive timeouts trip the breaker; the fourth call returns
    the fail-fast ``[peer unavailable]`` without waiting for ``timeout_s``."""
    bus = EventBus()
    await bus.start()
    try:
        target = PeerCapability(
            node_id="node.gone", reasoner="echo.specialist",
            tier="specialist", est_cost_ms=1.0,
        )
        # Long recovery so the breaker stays open for the fourth call.
        remote = RemoteReasoner(
            bus=bus, node_id="node.self", target=target,
            timeout_s=0.02, breaker_failure_threshold=3, breaker_recovery_s=30.0,
        )
        world = WorldModel(bus=bus)
        tools = ToolRegistry()

        for _ in range(3):
            r = await remote.answer(Intent(text="x"), world, tools)
            assert "timeout" in r.text.lower()

        assert remote._breaker.state == "open"

        # Fourth call: breaker is open, so we expect the fail-fast path and
        # it must return almost instantly rather than waiting for timeout_s.
        loop = asyncio.get_event_loop()
        t0 = loop.time()
        r = await remote.answer(Intent(text="x"), world, tools)
        elapsed = loop.time() - t0

        assert r.text == "[peer unavailable]"
        assert r.confidence == 0.0
        assert r.cost_ms == 0.0
        assert elapsed < remote.timeout_s  # fail-fast, not waiting on the bus
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_remote_reasoner_breaker_recovers_after_window_and_success() -> None:
    """After ``recovery_s`` elapses, a successful probe closes the breaker."""
    bus = EventBus()
    await bus.start()
    try:
        target = PeerCapability(
            node_id="node.peer", reasoner="echo.specialist",
            tier="specialist", est_cost_ms=1.0,
        )
        remote = RemoteReasoner(
            bus=bus, node_id="node.self", target=target,
            timeout_s=0.02, breaker_failure_threshold=3, breaker_recovery_s=0.05,
        )
        world = WorldModel(bus=bus)
        tools = ToolRegistry()

        # Drive the breaker open with three timeouts (no responder wired yet).
        for _ in range(3):
            await remote.answer(Intent(text="x"), world, tools)
        assert remote._breaker.state == "open"

        # Wait past recovery_s so the breaker transitions to half-open on call.
        await asyncio.sleep(0.08)

        # Now wire a responder so the probe succeeds and closes the breaker.
        async def on_request(e: Event) -> None:
            await bus.publish(
                Event(
                    topic="remote.reason.response",
                    origin="node.peer",
                    payload={
                        "request_id": e.payload["request_id"],
                        "text": "ok",
                        "confidence": 0.8,
                        "cost_ms": 1.0,
                    },
                )
            )
        bus.subscribe("remote.reason.request", on_request, name="test.recovery.responder")

        r = await remote.answer(Intent(text="probe"), world, tools)
        assert r.text == "ok"
        assert r.confidence == 0.8
        assert remote._breaker.state == "closed"
    finally:
        await bus.close()


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
