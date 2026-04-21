"""In-process async event bus.

Phase 1: asyncio-based pub/sub. Topic matching uses simple prefix wildcards
(e.g. "sense.*" matches "sense.audio.transcript").

Phase 2 drop-in: replace publish/subscribe internals with a NATS client.
Callers hold only the EventBus interface, so the swap is transparent.
"""
from __future__ import annotations

import asyncio
import fnmatch
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from .transport import InProcessTransport, Transport
from .types import Event

Handler = Callable[[Event], Awaitable[None]]


@dataclass
class Subscription:
    pattern: str
    handler: Handler
    name: str


@dataclass
class EventBus:
    transport: Transport = field(default_factory=InProcessTransport)
    _subs: list[Subscription] = field(default_factory=list)
    _tap: list[Event] = field(default_factory=list)  # rolling log for introspection
    _tap_limit: int = 1000

    def subscribe(self, pattern: str, handler: Handler, name: str = "") -> None:
        self._subs.append(Subscription(pattern=pattern, handler=handler, name=name or handler.__name__))

    async def start(self) -> None:
        """Activate the remote transport, if any."""
        await self.transport.start(self._deliver_local)

    async def _deliver_local(self, event: Event) -> None:
        """Entry point for events arriving from the transport. Fans to local
        subscribers but does NOT re-publish remotely (would cause loops)."""
        self._tap.append(event)
        if len(self._tap) > self._tap_limit:
            self._tap = self._tap[-self._tap_limit :]
        await self._dispatch(event)

    async def _dispatch(self, event: Event) -> None:
        tasks = [
            asyncio.create_task(s.handler(event))
            for s in self._subs
            if fnmatch.fnmatchcase(event.topic, s.pattern)
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def publish(self, event: Event) -> None:
        self._tap.append(event)
        if len(self._tap) > self._tap_limit:
            self._tap = self._tap[-self._tap_limit :]
        # Local fan-out + remote fan-out in parallel.
        await asyncio.gather(
            self._dispatch(event),
            self.transport.publish_remote(event),
            return_exceptions=True,
        )

    def recent(self, topic_pattern: str = "*", n: int = 50) -> list[Event]:
        matches = [e for e in self._tap if fnmatch.fnmatchcase(e.topic, topic_pattern)]
        return matches[-n:]

    async def close(self) -> None:
        await self.transport.close()
