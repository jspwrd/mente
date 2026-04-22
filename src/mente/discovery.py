"""Peer discovery + federated routing.

Each node announces its reasoners' capabilities on `meta.capability.announce`.
Other nodes subscribe, build a directory of remote specialists, and can
dispatch intents to them via `remote.reason.request` / `remote.reason.response`.

Phase 1: announcement is a periodic heartbeat. No leases, no conflict
resolution — last announcement wins.
Phase 2: signed capability manifests, health scoring, load balancing across
peers, trained routing from verifier feedback.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .bus import EventBus
from .logging import get_logger
from .reasoners import Reasoner
from .tools import ToolRegistry
from .types import Event, Intent, ReasonerTier, Response
from .world_model import WorldModel

_log = get_logger("discovery")


@dataclass
class PeerCapability:
    node_id: str
    reasoner: str
    tier: ReasonerTier
    est_cost_ms: float
    specialization: str = ""
    last_seen: float = 0.0


@dataclass
class Announcer:
    """Publishes local reasoners as capabilities on a timer."""
    bus: EventBus
    node_id: str
    reasoners: list[Reasoner]
    interval_s: float = 2.0
    specialization: str = ""

    async def announce_once(self) -> None:
        for r in self.reasoners:
            await self.bus.publish(
                Event(
                    topic="meta.capability.announce",
                    origin=self.node_id,
                    payload={
                        "node_id": self.node_id,
                        "reasoner": r.name,
                        "tier": r.tier,
                        "est_cost_ms": r.est_cost_ms,
                        "specialization": self.specialization,
                    },
                )
            )

    async def run(self, stop: asyncio.Event) -> None:
        # Announce immediately, then on a timer.
        await self.announce_once()
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.interval_s)
            except TimeoutError:
                await self.announce_once()


@dataclass
class Directory:
    """Remote capability directory. Updated from `meta.capability.announce`."""
    bus: EventBus
    self_node_id: str
    peers: dict[tuple[str, str], PeerCapability] = field(default_factory=dict)

    def wire(self) -> None:
        async def _on_announce(event: Event) -> None:
            p = event.payload
            try:
                peer_node = p["node_id"]
                reasoner_name = p["reasoner"]
            except KeyError:
                # Malformed announcement; ignore quietly.
                return
            if peer_node == self.self_node_id:
                return
            key = (peer_node, reasoner_name)
            self.peers[key] = PeerCapability(
                node_id=peer_node,
                reasoner=reasoner_name,
                tier=p.get("tier", "specialist"),
                est_cost_ms=p.get("est_cost_ms", 1000.0),
                specialization=p.get("specialization", ""),
                last_seen=event.ts,
            )
        self.bus.subscribe("meta.capability.announce", _on_announce, name="directory.ingest")

    def specialists(self) -> list[PeerCapability]:
        return [p for p in self.peers.values() if p.tier == "specialist"]


@dataclass
class _PendingFuture:
    """Tracks one in-flight remote request so `_prune_stale` can evict it."""
    future: asyncio.Future[dict[str, Any]]
    created_at: float


@dataclass
class RemoteReasoner:
    """Reasoner proxy that dispatches to a remote peer over the bus.

    `answer()` publishes a `remote.reason.request` with a unique request_id
    and awaits a `remote.reason.response` matching that id.

    Backpressure: `_pending` is a request-id keyed registry of futures; every
    exit path of `answer()` pops its own id, and `_prune_stale()` drops any
    futures older than `2 * timeout_s` on each call — so a peer that never
    responds cannot leak memory.
    """
    bus: EventBus
    node_id: str  # our node_id (for the origin field)
    target: PeerCapability
    timeout_s: float = 30.0
    _pending: dict[str, _PendingFuture] = field(default_factory=dict)
    _wired: bool = False

    @property
    def name(self) -> str:
        return f"remote:{self.target.node_id}:{self.target.reasoner}"

    @property
    def tier(self) -> ReasonerTier:
        return self.target.tier

    @property
    def est_cost_ms(self) -> float:
        return self.target.est_cost_ms + 20.0  # add transport overhead

    def _wire(self) -> None:
        if self._wired:
            return
        async def _on_response(event: Event) -> None:
            rid = event.payload.get("request_id")
            if not isinstance(rid, str):
                return
            entry = self._pending.pop(rid, None)
            if entry is not None and not entry.future.done():
                entry.future.set_result(event.payload)
        self.bus.subscribe("remote.reason.response", _on_response, name=f"remote.in.{self.target.node_id}")
        self._wired = True

    def _prune_stale(self) -> None:
        """Drop any pending futures older than `2 * timeout_s`.

        Defensive backstop for the rare case where `answer()`'s finally
        clause never runs (e.g. cancellation at an unusual moment) — keeps
        `_pending` bounded over long sessions.
        """
        cutoff = time.monotonic() - (2.0 * self.timeout_s)
        stale = [rid for rid, entry in self._pending.items() if entry.created_at < cutoff]
        for rid in stale:
            entry = self._pending.pop(rid, None)
            if entry is not None and not entry.future.done():
                entry.future.cancel()
        if stale:
            _log.info(
                "discovery.prune_stale",
                extra={"dropped": len(stale), "target": self.target.node_id},
            )

    async def answer(
        self, intent: Intent, world: WorldModel, tools: ToolRegistry
    ) -> Response:
        self._wire()
        self._prune_stale()
        request_id = uuid.uuid4().hex
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()
        self._pending[request_id] = _PendingFuture(future=fut, created_at=time.monotonic())
        await self.bus.publish(
            Event(
                topic="remote.reason.request",
                origin=self.node_id,
                payload={
                    "request_id": request_id,
                    "target_node": self.target.node_id,
                    "target_reasoner": self.target.reasoner,
                    "intent": intent.text,
                },
            )
        )
        try:
            try:
                data = await asyncio.wait_for(fut, timeout=self.timeout_s)
            except TimeoutError:
                return Response(
                    text=f"[timeout from {self.target.node_id}]",
                    reasoner=self.name, tier=self.tier,
                    confidence=0.0, cost_ms=self.timeout_s * 1000,
                )
        finally:
            # Cover every exit (resolved, timeout, cancel, exception) so a
            # leaked future can never accumulate here.
            self._pending.pop(request_id, None)
        return Response(
            text=data.get("text", ""),
            reasoner=self.name, tier=self.tier,
            confidence=data.get("confidence", 0.5),
            cost_ms=data.get("cost_ms", 0.0),
        )


@dataclass
class RemoteRequestHandler:
    """Peer side of federated routing.

    Subscribes to `remote.reason.request`, dispatches to local reasoners
    matching `target_reasoner`, and publishes `remote.reason.response`.
    """
    bus: EventBus
    node_id: str
    reasoners: list[Reasoner]
    world: WorldModel
    tools: ToolRegistry

    def wire(self) -> None:
        async def _on_request(event: Event) -> None:
            if event.payload.get("target_node") != self.node_id:
                return
            target_name = event.payload.get("target_reasoner")
            reasoner = next((r for r in self.reasoners if r.name == target_name), None)
            if reasoner is None:
                await self.bus.publish(
                    Event(
                        topic="remote.reason.response",
                        origin=self.node_id,
                        payload={
                            "request_id": event.payload.get("request_id"),
                            "text": f"[unknown reasoner {target_name}]",
                            "confidence": 0.0,
                            "cost_ms": 0.0,
                        },
                    )
                )
                return
            intent = Intent(text=event.payload.get("intent", ""))
            response = await reasoner.answer(intent, self.world, self.tools)
            await self.bus.publish(
                Event(
                    topic="remote.reason.response",
                    origin=self.node_id,
                    payload={
                        "request_id": event.payload.get("request_id"),
                        "text": response.text,
                        "confidence": response.confidence,
                        "cost_ms": response.cost_ms,
                    },
                )
            )
        self.bus.subscribe("remote.reason.request", _on_request, name=f"remote.handler.{self.node_id}")
