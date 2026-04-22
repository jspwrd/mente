"""End-to-end multi-listener federation tests.

Spins up one hub + two spokes in-process over real TCP loopback (three
`EventBus` instances, three `TCPTransport`s on distinct ephemeral ports — hub
binds the hub port, each spoke connects to it). Each node runs `Announcer` +
`Directory` + `RemoteRequestHandler`, so the full discovery / dispatch path is
exercised without any subprocess.

This complements the single-peer subprocess test in `test_federated_cli.py`
and the pure in-process unit tests in `test_discovery.py`: here we check that
capability routing *and* failure tolerance work with more than one spoke.
"""
from __future__ import annotations

import asyncio
import socket
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from mente.bus import EventBus
from mente.discovery import (
    Announcer,
    Directory,
    RemoteRequestHandler,
)
from mente.tools import ToolRegistry
from mente.transport import TCPTransport
from mente.types import Event, Intent, ReasonerTier, Response
from mente.world_model import WorldModel


def _pick_free_port() -> int:
    """Bind `:0`, read the OS-assigned port, release it. Good-enough race-wise."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@dataclass
class NamedReasoner:
    """Reasoner that records every intent it handles, with a unique name.

    We need distinct names per spoke so `target_reasoner` dispatch can select
    one and not the other — the stock `EchoReasoner` always uses
    `echo.specialist` which would collide.
    """
    name: str
    handled: list[str]
    tier: ReasonerTier = "specialist"
    est_cost_ms: float = 1.0

    async def answer(self, intent: Intent, world: WorldModel, tools: ToolRegistry) -> Response:
        self.handled.append(intent.text)
        return Response(
            text=f"{self.name}:{intent.text}",
            reasoner=self.name,
            tier=self.tier,
            confidence=0.9,
            cost_ms=self.est_cost_ms,
        )


@dataclass
class Node:
    """One federated node: bus + directory + announcer + handler."""
    node_id: str
    bus: EventBus
    directory: Directory
    announcer: Announcer
    handler: RemoteRequestHandler
    reasoner: NamedReasoner


async def _make_node(
    node_id: str, port: int, role: str, reasoner_name: str, specialization: str
) -> Node:
    """Build one fully-wired federated node (bus + directory + announcer + handler)."""
    bus = EventBus(transport=TCPTransport(node_id=node_id, port=port, role=role))
    await bus.start()
    reasoner = NamedReasoner(name=reasoner_name, handled=[])
    directory = Directory(bus=bus, self_node_id=node_id)
    directory.wire()
    announcer = Announcer(
        bus=bus, node_id=node_id,
        reasoners=[reasoner], specialization=specialization,
    )
    handler = RemoteRequestHandler(
        bus=bus, node_id=node_id,
        reasoners=[reasoner],
        world=WorldModel(bus=bus),
        tools=ToolRegistry(),
    )
    handler.wire()
    return Node(
        node_id=node_id, bus=bus,
        directory=directory, announcer=announcer,
        handler=handler, reasoner=reasoner,
    )


async def _wait_for_hub_writers(hub: Node, expected: int, deadline_s: float = 2.0) -> None:
    """Spin until the hub's accept-loop has registered `expected` spoke writers.

    `asyncio.open_connection` on the spoke returns before the hub's `_accept`
    callback has run, so immediately calling `announce_once` can race the
    hub's fanout list. Wait explicitly for both writers to land.
    """
    loop = asyncio.get_running_loop()
    stop = loop.time() + deadline_s
    while loop.time() < stop:
        if len(hub.bus.transport._writers) >= expected:  # type: ignore[attr-defined]
            return
        await asyncio.sleep(0.01)
    raise AssertionError(
        f"hub did not register {expected} writers in {deadline_s}s"
    )


@pytest.fixture
async def three_node_cluster() -> AsyncIterator[tuple[Node, Node, Node]]:
    """Hub + spoke1 + spoke2, all wired and announced once. Torn down on exit."""
    port = _pick_free_port()
    hub = await _make_node("hub", port, "hub", "specialist.coordinator", "coordinator")
    spoke1 = await _make_node("spoke1", port, "spoke", "specialist.math", "math")
    spoke2 = await _make_node("spoke2", port, "spoke", "specialist.code", "code")
    try:
        # Wait for the hub's accept-loop to register both spokes before
        # announcing — otherwise the hub's fanout list is still empty when
        # `announce_once` runs and spokes never see the hub's capability.
        await _wait_for_hub_writers(hub, expected=2)
        # One announcement round from each — directory is populated after
        # this because TCPTransport fanout runs inside `publish`.
        await asyncio.gather(
            hub.announcer.announce_once(),
            spoke1.announcer.announce_once(),
            spoke2.announcer.announce_once(),
        )
        # Give the hub's fanout and each spoke's listener a tick to drain.
        await asyncio.sleep(0.2)
        yield hub, spoke1, spoke2
    finally:
        # Close spokes first so the hub's accept-loop sees clean EOFs.
        await asyncio.gather(
            spoke1.bus.close(),
            spoke2.bus.close(),
            return_exceptions=True,
        )
        await hub.bus.close()


@pytest.mark.asyncio
async def test_announcements_reach_hub_from_both_spokes(
    three_node_cluster: tuple[Node, Node, Node],
) -> None:
    """Hub's `Directory` learns both spokes' reasoners after one announce round."""
    hub, spoke1, spoke2 = three_node_cluster

    peer_ids = {p.node_id for p in hub.directory.peers.values()}
    assert "spoke1" in peer_ids, f"hub did not learn spoke1: {hub.directory.peers!r}"
    assert "spoke2" in peer_ids, f"hub did not learn spoke2: {hub.directory.peers!r}"

    # Each spoke also learns the other via the hub's rebroadcast.
    assert ("hub", "specialist.coordinator") in spoke1.directory.peers
    assert ("hub", "specialist.coordinator") in spoke2.directory.peers


@pytest.mark.asyncio
async def test_request_routes_to_correct_spoke(
    three_node_cluster: tuple[Node, Node, Node],
) -> None:
    """A request targeted at `spoke1` is handled only by spoke1."""
    hub, spoke1, spoke2 = three_node_cluster

    # Hub publishes a remote.reason.request naming spoke1 as the target. The
    # request travels over TCP to both spokes; only spoke1's handler matches
    # because `RemoteRequestHandler` checks `target_node == self.node_id`.
    await hub.bus.publish(
        Event(
            topic="remote.reason.request",
            origin="hub",
            payload={
                "request_id": "req-multi-1",
                "target_node": "spoke1",
                "target_reasoner": spoke1.reasoner.name,
                "intent": "compute 2+2",
            },
        )
    )
    # Let the fanout + handler round-trip settle.
    await asyncio.sleep(0.3)

    assert spoke1.reasoner.handled == ["compute 2+2"], (
        f"spoke1 should have handled exactly one request, got {spoke1.reasoner.handled!r}"
    )
    assert spoke2.reasoner.handled == [], (
        f"spoke2 should not have touched the request, got {spoke2.reasoner.handled!r}"
    )


@pytest.mark.asyncio
async def test_one_spoke_disappearing_does_not_break_other(
    three_node_cluster: tuple[Node, Node, Node],
) -> None:
    """Close spoke1's bus; spoke2 must still answer a targeted request."""
    hub, spoke1, spoke2 = three_node_cluster

    await spoke1.bus.close()
    # Let the hub notice the half-open socket via the next fanout write.
    await asyncio.sleep(0.1)

    await hub.bus.publish(
        Event(
            topic="remote.reason.request",
            origin="hub",
            payload={
                "request_id": "req-multi-2",
                "target_node": "spoke2",
                "target_reasoner": spoke2.reasoner.name,
                "intent": "still alive?",
            },
        )
    )
    await asyncio.sleep(0.3)

    assert spoke2.reasoner.handled == ["still alive?"], (
        f"spoke2 should still serve after spoke1 went away; got {spoke2.reasoner.handled!r}"
    )
    # spoke1 is closed — its reasoner should not have observed anything.
    assert spoke1.reasoner.handled == []
