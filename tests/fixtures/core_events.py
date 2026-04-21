"""Event-capture helper for bus tests."""
from __future__ import annotations

from dataclasses import dataclass, field

from aria.types import Event


@dataclass
class EventCapture:
    """Collects events delivered to a subscribed handler."""
    events: list[Event] = field(default_factory=list)

    async def handler(self, event: Event) -> None:
        self.events.append(event)

    def topics(self) -> list[str]:
        return [e.topic for e in self.events]

    def clear(self) -> None:
        self.events.clear()
