"""Curiosity loop — idle-time self-prompting.

When the system has been idle for a while, the curiosity loop inspects the
world model and latent state for gaps and generates synthetic intents that
fill them in. Those intents run through the normal routing/reasoning path;
results flow back into memory and the world model.

Phase 1 heuristics:
  - If the user has notes but no summary belief, ask "what do I remember?"
  - If the latent digest has a low accept_rate, flag it.
  - If an entity has only one attribute, ask about related attributes.

Phase 2: a learned intrinsic-objective model (§3 of the advanced design)
generating hypotheses about what's worth knowing, scored by information
gain and empowerment.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from .bus import EventBus
from .state import LatentState
from .types import Event, Intent
from .world_model import WorldModel


@dataclass
class Curiosity:
    bus: EventBus
    world: WorldModel
    latent: LatentState
    idle_threshold_s: float = 5.0
    interval_s: float = 3.0
    _last_user_intent_ts: float = field(default_factory=time.time)
    _generated: set[str] = field(default_factory=set)

    def wire(self) -> None:
        async def _on_intent(event: Event) -> None:
            if event.origin not in ("curiosity", "self"):
                self._last_user_intent_ts = time.time()
        self.bus.subscribe("intent.*", _on_intent, name="curiosity.track_activity")

    def _gaps(self) -> list[str]:
        """Return a list of curiosity-worthy questions based on current state."""
        gaps: list[str] = []

        digest = self.latent.get("last_digest") or {}
        accept = digest.get("accept_rate")
        if accept is not None and accept < 0.7:
            gaps.append(f"why is my accept rate only {accept}?")

        note_count = digest.get("note_count", 0)
        if note_count > 0:
            gaps.append("what do you remember?")

        for ent in sorted(self.world.entities()):
            attrs = list(self.world.entity(ent).keys())
            if len(attrs) == 1:
                gaps.append(f"what do you know about {ent}?")

        # Dedupe against already-asked-this-session.
        fresh = [g for g in gaps if g not in self._generated]
        for g in fresh:
            self._generated.add(g)
        return fresh

    async def tick(self) -> list[str]:
        """One curiosity cycle. Returns the intents emitted (for testing)."""
        now = time.time()
        if now - self._last_user_intent_ts < self.idle_threshold_s:
            return []
        emitted: list[str] = []
        for text in self._gaps():
            await self.bus.publish(
                Event(
                    topic="curiosity.generate",
                    origin="curiosity",
                    payload={"text": text},
                )
            )
            emitted.append(text)
        return emitted

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.interval_s)
            except asyncio.TimeoutError:
                await self.tick()
