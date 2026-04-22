"""In-process async event bus.

Phase 1: asyncio-based pub/sub. Topic matching uses simple prefix wildcards
(e.g. "sense.*" matches "sense.audio.transcript").

Phase 2 drop-in: replace publish/subscribe internals with a NATS client.
Callers hold only the EventBus interface, so the swap is transparent.
"""
from __future__ import annotations

import asyncio
import fnmatch
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from .logging import get_logger
from .transport import InProcessTransport, Transport
from .types import Event

# Handler must be a coroutine function (``async def``), not a generic
# awaitable-returning callable — ``asyncio.create_task`` requires a real
# Coroutine, and narrowing the type here lets strict mypy verify it.
Handler = Callable[[Event], Coroutine[Any, Any, None]]

_log = get_logger("bus")


@dataclass
class Subscription:
    """A registered handler plus the topic pattern that selects events for it.

    Attributes:
        pattern: ``fnmatch``-style topic pattern (e.g. ``"sense.*"``).
        handler: Async callable invoked for each matching event.
        name: Human-readable label, used in logs and introspection.
    """
    pattern: str
    handler: Handler
    name: str


@dataclass
class EventBus:
    """Async pub/sub bus with pluggable transport and a rolling tap log.

    Subscribers register a pattern and a handler; publishing dispatches the
    event locally to every matching subscriber and, in parallel, hands the
    event to the transport for remote delivery. A bounded tap buffer keeps
    recent events around for introspection and tests.

    Attributes:
        transport: Backing transport. Defaults to in-process (loop-local);
            swap for a NATS-backed transport to fan out across nodes.
        _tap_limit: Maximum number of events retained in the tap buffer.
    """
    transport: Transport = field(default_factory=InProcessTransport)
    _subs: list[Subscription] = field(default_factory=list)
    _tap: list[Event] = field(default_factory=list)  # rolling log for introspection
    _tap_limit: int = 1000

    def subscribe(self, pattern: str, handler: Handler, name: str = "") -> None:
        """Register ``handler`` to receive events whose topic matches ``pattern``.

        Patterns use ``fnmatch`` semantics, so ``"sense.*"`` matches
        ``"sense.audio"`` but not ``"sense.audio.transcript"`` — use
        ``"sense.*.*"`` or ``"sense.**"``-style patterns as needed.

        Args:
            pattern: Topic pattern to match against ``event.topic``.
            handler: Async callable invoked with each matching event.
            name: Optional label for logs; defaults to ``handler.__name__``.
        """
        sub_name = name or handler.__name__
        self._subs.append(Subscription(pattern=pattern, handler=handler, name=sub_name))
        # DEBUG: Runtime wiring subscribes many times at startup; keep it quiet at INFO.
        _log.debug("subscribe %s -> %s", pattern, sub_name)

    async def start(self) -> None:
        """Activate the remote transport, if any.

        For the in-process transport this is a no-op. For a real transport
        (e.g. NATS) it opens the underlying connection and wires incoming
        remote events back to the local dispatch path.
        """
        await self.transport.start(self._deliver_local)

    async def _deliver_local(self, event: Event) -> None:
        """Entry point for events arriving from the transport. Fans to local
        subscribers but does NOT re-publish remotely (would cause loops)."""
        self._tap.append(event)
        if len(self._tap) > self._tap_limit:
            self._tap = self._tap[-self._tap_limit :]
        await self._dispatch(event)

    async def _dispatch(self, event: Event) -> None:
        tasks: list[asyncio.Task[None]] = [
            asyncio.create_task(s.handler(event))
            for s in self._subs
            if fnmatch.fnmatchcase(event.topic, s.pattern)
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def publish(self, event: Event) -> None:
        """Fan out ``event`` to local subscribers and the remote transport.

        The call appends the event to the tap buffer (trimming to
        ``_tap_limit``), then runs two fan-outs in parallel: local pattern
        matching dispatches to each matching subscriber as its own task,
        and the transport ships the event off-node. Exceptions from either
        side are collected rather than raised — a single broken handler
        must not take down the bus.

        Args:
            event: The event to publish. Its ``topic`` is matched against
                every subscriber pattern via ``fnmatch``.
        """
        self._tap.append(event)
        if len(self._tap) > self._tap_limit:
            self._tap = self._tap[-self._tap_limit :]
        # Hot path: gate on isEnabledFor so the ``extra=`` dict isn't built
        # when DEBUG is off. ``logger.debug`` checks level internally too, but
        # only after evaluating call arguments.
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(
                "publish %s origin=%s",
                event.topic,
                event.origin,
                extra={"trace_id": event.trace_id},
            )
        # Local fan-out + remote fan-out in parallel.
        await asyncio.gather(
            self._dispatch(event),
            self.transport.publish_remote(event),
            return_exceptions=True,
        )

    def recent(self, topic_pattern: str = "*", n: int = 50) -> list[Event]:
        """Return the most recent tapped events matching ``topic_pattern``.

        Useful for tests and introspection. The returned list is a fresh
        slice; mutating it will not affect the bus.

        Args:
            topic_pattern: ``fnmatch``-style pattern to filter tap entries.
            n: Maximum number of events to return (most recent last).

        Returns:
            A list of matching events in chronological order, at most
            ``n`` long.
        """
        matches = [e for e in self._tap if fnmatch.fnmatchcase(e.topic, topic_pattern)]
        return matches[-n:]

    async def close(self) -> None:
        """Tear down the transport. Safe to call multiple times."""
        await self.transport.close()
