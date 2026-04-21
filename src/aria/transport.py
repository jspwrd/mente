"""Distributed bus transport.

Transport is the seam between the in-process EventBus and any remote peers.
Phase 1: the bus was process-local. Phase 3: a Transport can fan an event
out to peer processes and deliver peer events back into the local bus.

- InProcessTransport: no-op. Default.
- TCPTransport: JSON-lines protocol over asyncio.Streams. Exactly one node
  binds as the hub; others connect as spokes. The hub rebroadcasts each
  received event to every other spoke, so every peer sees every remote event
  exactly once.

A NATS backend would plug in here as a third Transport without changing any
caller. Same for Redis streams, ZeroMQ, etc.
"""
from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

from .types import Event

RemoteHandler = Callable[[Event], Awaitable[None]]


class Transport(Protocol):
    async def start(self, on_remote: RemoteHandler) -> None: ...
    async def publish_remote(self, event: Event) -> None: ...
    async def close(self) -> None: ...


@dataclass
class InProcessTransport:
    """No-op transport. Matches the Protocol; does nothing."""
    async def start(self, on_remote: RemoteHandler) -> None:
        return None

    async def publish_remote(self, event: Event) -> None:
        return None

    async def close(self) -> None:
        return None


def _encode(event: Event) -> bytes:
    payload = {
        "topic": event.topic,
        "payload": event.payload,
        "origin": event.origin,
        "trace_id": event.trace_id,
        "ts": event.ts,
        "confidence": event.confidence,
    }
    return (json.dumps(payload, default=str) + "\n").encode()


def _decode(line: bytes) -> Event:
    d = json.loads(line.decode())
    return Event(
        topic=d["topic"],
        payload=d.get("payload") or {},
        origin=d["origin"],
        trace_id=d.get("trace_id", ""),
        ts=d.get("ts", 0.0),
        confidence=d.get("confidence", 1.0),
    )


@dataclass
class TCPTransport:
    """JSON-line bus over TCP.

    Role:
      - hub: binds the port, accepts spokes, rebroadcasts incoming events
        to every other spoke.
      - spoke: connects to the hub, sends local publishes, receives remote
        events.

    The node_id stamps outbound events so a node can ignore its own
    echoes if the hub rebroadcasts them.
    """
    node_id: str
    host: str = "127.0.0.1"
    port: int = 7722
    role: str = "spoke"  # "hub" or "spoke"

    _server: asyncio.base_events.Server | None = None
    _writers: list[asyncio.StreamWriter] = field(default_factory=list)
    _hub_reader: asyncio.StreamReader | None = None
    _hub_writer: asyncio.StreamWriter | None = None
    _on_remote: RemoteHandler | None = None
    _listen_tasks: list[asyncio.Task] = field(default_factory=list)

    async def start(self, on_remote: RemoteHandler) -> None:
        self._on_remote = on_remote
        if self.role == "hub":
            await self._start_hub()
        else:
            await self._start_spoke()

    async def _start_hub(self) -> None:
        async def _accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            self._writers.append(writer)
            try:
                while line := await reader.readline():
                    try:
                        event = _decode(line)
                    except Exception:
                        continue
                    # Rebroadcast to every peer except the sender, then
                    # deliver locally.
                    await self._fanout(event, exclude=writer)
                    if self._on_remote:
                        await self._on_remote(event)
            finally:
                if writer in self._writers:
                    self._writers.remove(writer)
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass

        self._server = await asyncio.start_server(_accept, self.host, self.port)

    async def _fanout(self, event: Event, exclude: asyncio.StreamWriter | None = None) -> None:
        buf = _encode(event)
        stale: list[asyncio.StreamWriter] = []
        for w in self._writers:
            if w is exclude:
                continue
            try:
                w.write(buf)
                await w.drain()
            except (ConnectionResetError, BrokenPipeError):
                stale.append(w)
        for w in stale:
            if w in self._writers:
                self._writers.remove(w)

    async def _start_spoke(self) -> None:
        self._hub_reader, self._hub_writer = await asyncio.open_connection(self.host, self.port)

        async def _listen() -> None:
            assert self._hub_reader is not None
            while line := await self._hub_reader.readline():
                try:
                    event = _decode(line)
                except Exception:
                    continue
                # Skip our own echoes.
                if event.origin == self.node_id:
                    continue
                if self._on_remote:
                    await self._on_remote(event)

        self._listen_tasks.append(asyncio.create_task(_listen()))

    async def publish_remote(self, event: Event) -> None:
        if self.role == "hub":
            await self._fanout(event)
        else:
            if self._hub_writer is None:
                return
            try:
                self._hub_writer.write(_encode(event))
                await self._hub_writer.drain()
            except (ConnectionResetError, BrokenPipeError):
                pass

    async def close(self) -> None:
        for t in self._listen_tasks:
            t.cancel()
        if self._hub_writer:
            self._hub_writer.close()
            try:
                await self._hub_writer.wait_closed()
            except Exception:
                pass
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        for w in list(self._writers):
            w.close()
        self._writers.clear()


def tcp_from_env(node_id: str) -> Transport:
    """Build a TCPTransport from env vars (ARIA_BUS_HOST, ARIA_BUS_PORT,
    ARIA_BUS_ROLE). Falls back to InProcessTransport if unset."""
    if not os.environ.get("ARIA_BUS_ROLE"):
        return InProcessTransport()
    return TCPTransport(
        node_id=node_id,
        host=os.environ.get("ARIA_BUS_HOST", "127.0.0.1"),
        port=int(os.environ.get("ARIA_BUS_PORT", "7722")),
        role=os.environ.get("ARIA_BUS_ROLE", "spoke"),
    )
