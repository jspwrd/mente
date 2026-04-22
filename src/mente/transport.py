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

Optional HMAC handshake (`auth_secret`): when set on both hub and spoke, the
spoke sends `AUTH <hex(hmac-sha256(secret, node_id + ts))>` as its first
line, and the hub verifies with `hmac.compare_digest`. Mismatches drop the
connection. When `auth_secret is None` on either side the handshake is
skipped entirely — preserves the default inproc/demo experience.
"""
from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Protocol

from .logging import get_logger
from .types import Event

RemoteHandler = Callable[[Event], Awaitable[None]]

_log = get_logger("transport")

# Tolerance window for handshake timestamps — guards against replay drift while
# staying generous enough for slow CI hosts.
_AUTH_MAX_SKEW_S: float = 60.0
# Line length cap for the handshake frame so a rogue client can't exhaust
# memory by shovelling bytes without a newline.
_AUTH_MAX_LINE: int = 4096


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


def _hmac_token(secret: str, node_id: str, ts: float) -> str:
    """Deterministic HMAC-SHA256 over `node_id|ts`, hex-encoded."""
    mac = hmac.new(
        secret.encode(),
        f"{node_id}|{ts}".encode(),
        sha256,
    )
    return mac.hexdigest()


def _build_auth_line(secret: str, node_id: str) -> bytes:
    ts = time.time()
    token = _hmac_token(secret, node_id, ts)
    return f"AUTH {node_id} {ts} {token}\n".encode()


def _verify_auth_line(secret: str, line: bytes) -> str | None:
    """Return the authenticated `node_id` if the frame is valid, else None."""
    try:
        text = line.decode().strip()
    except UnicodeDecodeError:
        return None
    parts = text.split(" ")
    if len(parts) != 4 or parts[0] != "AUTH":
        return None
    _, node_id, ts_s, token = parts
    try:
        ts = float(ts_s)
    except ValueError:
        return None
    if abs(time.time() - ts) > _AUTH_MAX_SKEW_S:
        return None
    expected = _hmac_token(secret, node_id, ts)
    if not hmac.compare_digest(expected, token):
        return None
    return node_id


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

    `auth_secret` (optional): when set, the spoke sends an HMAC handshake
    as the first line on every connection and the hub validates it before
    accepting any events. Unset on either side disables the handshake.
    """
    node_id: str
    host: str = "127.0.0.1"
    port: int = 7722
    role: str = "spoke"  # "hub" or "spoke"
    auth_secret: str | None = None

    _server: asyncio.base_events.Server | None = None
    _writers: list[asyncio.StreamWriter] = field(default_factory=list)
    _hub_reader: asyncio.StreamReader | None = None
    _hub_writer: asyncio.StreamWriter | None = None
    _on_remote: RemoteHandler | None = None
    _listen_tasks: list[asyncio.Task[None]] = field(default_factory=list)

    async def start(self, on_remote: RemoteHandler) -> None:
        self._on_remote = on_remote
        if self.role == "hub":
            await self._start_hub()
        else:
            await self._start_spoke()

    async def _handshake_hub(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> bool:
        """Read the spoke's AUTH line and verify it. Returns True on success.

        On any failure the writer is closed and False is returned — callers
        must return without registering the writer.
        """
        if self.auth_secret is None:
            return True
        peer = writer.get_extra_info("peername")
        reason: str
        authed: str | None = None
        try:
            first = await reader.readline()
        except (ConnectionError, BrokenPipeError, OSError):
            reason = "read-failed"
        else:
            if not first or len(first) > _AUTH_MAX_LINE:
                reason = "empty-or-oversize"
            else:
                authed = _verify_auth_line(self.auth_secret, first)
                reason = "bad-hmac"
        if authed is not None:
            _log.info("transport.auth.accept", extra={"peer": str(peer), "node_id": authed})
            return True
        _log.info("transport.auth.reject", extra={"peer": str(peer), "reason": reason})
        with contextlib.suppress(Exception):
            writer.close()
            await writer.wait_closed()
        return False

    async def _start_hub(self) -> None:
        async def _accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            peer = writer.get_extra_info("peername")
            if not await self._handshake_hub(reader, writer):
                return
            self._writers.append(writer)
            try:
                while line := await reader.readline():
                    try:
                        event = _decode(line)
                    except (ValueError, KeyError, json.JSONDecodeError):
                        # Malformed frame: drop just this message.
                        continue
                    # Rebroadcast to every peer except the sender, then
                    # deliver locally.
                    await self._fanout(event, exclude=writer)
                    if self._on_remote:
                        await self._on_remote(event)
            except (ConnectionError, BrokenPipeError, OSError) as exc:
                _log.info(
                    "transport.peer.drop",
                    extra={"peer": str(peer), "reason": type(exc).__name__},
                )
            finally:
                if writer in self._writers:
                    self._writers.remove(writer)
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()

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

        # Send handshake before any publishes hit the wire.
        if self.auth_secret is not None:
            self._hub_writer.write(_build_auth_line(self.auth_secret, self.node_id))
            try:
                await self._hub_writer.drain()
            except (ConnectionError, BrokenPipeError) as exc:
                _log.info(
                    "transport.auth.send-failed",
                    extra={"node_id": self.node_id, "reason": type(exc).__name__},
                )
                raise

        async def _listen() -> None:
            assert self._hub_reader is not None
            while line := await self._hub_reader.readline():
                try:
                    event = _decode(line)
                except (ValueError, KeyError, json.JSONDecodeError):
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
            with contextlib.suppress(Exception):
                await self._hub_writer.wait_closed()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        for w in list(self._writers):
            w.close()
        self._writers.clear()


def tcp_from_env(node_id: str) -> Transport:
    """Build a TCPTransport from env vars (MENTE_BUS_HOST, MENTE_BUS_PORT,
    MENTE_BUS_ROLE, MENTE_BUS_SECRET). Falls back to InProcessTransport if
    role is unset."""
    if not os.environ.get("MENTE_BUS_ROLE"):
        return InProcessTransport()
    return TCPTransport(
        node_id=node_id,
        host=os.environ.get("MENTE_BUS_HOST", "127.0.0.1"),
        port=int(os.environ.get("MENTE_BUS_PORT", "7722")),
        role=os.environ.get("MENTE_BUS_ROLE", "spoke"),
        auth_secret=os.environ.get("MENTE_BUS_SECRET") or None,
    )
