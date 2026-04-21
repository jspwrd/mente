"""World model: the shared blackboard of 'what the system believes is true now'.

Entity-attribute-value store with TTLs and confidence. Every write emits a
'state.*' event so subscribers react to changes instead of polling.

Phase 2 drop-in: swap the dict backend for Redis/Dragonfly; add a learned
differentiable world-model surface that predicts next-state given action.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .bus import EventBus
from .types import Belief, Event


@dataclass
class WorldModel:
    bus: EventBus
    _beliefs: dict[tuple[str, str], Belief] = field(default_factory=dict)

    async def assert_belief(self, b: Belief) -> None:
        key = (b.entity, b.attribute)
        prev = self._beliefs.get(key)
        self._beliefs[key] = b
        await self.bus.publish(
            Event(
                topic=f"state.{b.entity}.{b.attribute}",
                origin="world_model",
                confidence=b.confidence,
                payload={
                    "entity": b.entity,
                    "attribute": b.attribute,
                    "value": b.value,
                    "previous": prev.value if prev else None,
                },
            )
        )

    def get(self, entity: str, attribute: str) -> Belief | None:
        b = self._beliefs.get((entity, attribute))
        if b is None or not b.is_live():
            return None
        return b

    def entity(self, entity: str) -> dict[str, Any]:
        return {
            attr: b.value
            for (ent, attr), b in self._beliefs.items()
            if ent == entity and b.is_live()
        }

    def snapshot(self) -> list[Belief]:
        return [b for b in self._beliefs.values() if b.is_live()]

    def entities(self) -> Iterable[str]:
        return {ent for (ent, _), b in self._beliefs.items() if b.is_live()}
