"""Event-loop runtime.

This is the piece that makes the system a *process* rather than a function.
It owns the bus, the world model, the memory, the reasoners, and a long-lived
asyncio loop. Inputs arrive as events (intent.*, sense.*, tool.*, timer.*)
and are routed, reasoned, verified, and emitted as responses.

Setup is split into six phases (invoked in order by ``__post_init__``):
``_setup_logging_and_storage`` wires the logger, on-disk state and verifier;
``_setup_reasoners`` picks the default roster; ``_setup_router`` builds the
metacog + router; ``_setup_tools_and_subscribers`` registers builtin tools
and persistence hooks; ``_setup_self_model`` attaches the self-reflection
answerer; ``_setup_background_surfaces`` prepares the consolidator, curiosity
loop and the curiosity→intent bridge. Read any phase in isolation to find
what you need.

Phase 1: single-process, single-agent, in-memory bus.
Phase 2: multi-process via NATS; multiple agent peers sharing the bus; idle
timers that trigger consolidation and curiosity-driven self-prompting.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bus import EventBus
from .config import MenteConfig
from .consolidator import Consolidator
from .curiosity import Curiosity
from .embeddings import SemanticMemory
from .logging import configure as configure_logging
from .logging import get_logger
from .memory import FastMemory, SlowMemory
from .metacog import Metacog
from .reasoners import DeepSimulatedReasoner, FastHeuristicReasoner, Reasoner, set_self_model_hook
from .router import Router
from .self_model import SelfModel
from .specialists import CodeSpecialist
from .state import LatentState
from .synthesis import LibraryStore, SynthesisReasoner
from .tools import ToolRegistry
from .types import Event, Intent, Response
from .verifier import Verifier
from .world_model import WorldModel

_log = get_logger("runtime")


@dataclass
class Runtime:
    """Top-level process object wiring the bus, memories, and reasoners.

    A ``Runtime`` owns everything needed to turn an :class:`Intent` into a
    :class:`Response`: an event bus, a world model, fast/slow/semantic
    memories, a tool registry, a reasoner roster, a router, a verifier, a
    self-model, and idle-time consolidation + curiosity loops. Construction
    seeds default reasoners (using :class:`AnthropicReasoner` when an API
    key is available, otherwise :class:`DeepSimulatedReasoner`), registers
    the default tool set, and wires subscribers that persist ``state.*``
    and ``response.*`` events to slow memory.

    Attributes:
        root: Filesystem directory where episodic SQLite, semantic SQLite,
            latent state, and the synthesis library are stored. Created if
            it does not exist.
        node_id: Identifier for this runtime instance; surfaces in logs
            and (Phase 2) in remote bus messages.
        config: Tunables (log level, verifier threshold, router trade-off,
            consolidation/curiosity intervals). Defaults to
            :meth:`MenteConfig.default`.
        bus: The event bus. Replace to swap transports.
        world: World model; constructed from ``bus`` in ``__post_init__``.
        fast_mem: In-process scratchpad.
        slow_mem: Episodic SQLite log rooted at ``root``.
        semantic_mem: Vector-search memory rooted at ``root``.
        latent: Persisted latent summary state at ``root/latent.json``.
        tools: Tool registry; pre-populated with clock + memory tools.
        reasoners: Ordered reasoner roster considered by the router.
        router: Dispatcher picking a reasoner per intent.
        verifier: Post-hoc acceptance check on every response.
        self_model: Answers self-referential queries (wired as a hook
            into :class:`FastHeuristicReasoner`).
        consolidator: Background task summarising slow memory.
        curiosity: Background task that self-prompts when idle.
    """
    root: Path
    node_id: str = "mente.local"
    config: MenteConfig = field(default_factory=MenteConfig.default)
    bus: EventBus = field(default_factory=EventBus)
    world: WorldModel = field(init=False)
    fast_mem: FastMemory = field(default_factory=FastMemory)
    slow_mem: SlowMemory = field(init=False)
    semantic_mem: SemanticMemory = field(init=False)
    latent: LatentState = field(init=False)
    tools: ToolRegistry = field(default_factory=ToolRegistry)
    reasoners: list[Reasoner] = field(default_factory=list)
    router: Router = field(init=False)
    verifier: Verifier = field(init=False)
    self_model: SelfModel = field(init=False)
    consolidator: Consolidator = field(init=False)
    curiosity: Curiosity = field(init=False)
    _consolidator_stop: asyncio.Event = field(default_factory=asyncio.Event)
    _curiosity_stop: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self) -> None:
        self._setup_logging_and_storage()
        self._setup_reasoners()
        self._setup_router()
        self._setup_tools_and_subscribers()
        self._setup_self_model()
        self._setup_background_surfaces()

    # -- setup phases -------------------------------------------------------
    def _setup_logging_and_storage(self) -> None:
        """Configure logging, create the state dir, build on-disk stores."""
        configure_logging(level=self.config.log_level, json=self.config.log_json)
        self.root.mkdir(parents=True, exist_ok=True)
        self.world = WorldModel(bus=self.bus)
        self.slow_mem = SlowMemory(db_path=self.root / "episodic.sqlite")
        self.semantic_mem = SemanticMemory(db_path=self.root / "semantic.sqlite")
        self.latent = LatentState.load(self.root / "latent.json")
        self.verifier = Verifier(min_confidence=self.config.verifier_min_confidence)
        self.library = LibraryStore(path=self.root / "library.json")
        _log.info("runtime initialized", extra={"node_id": self.node_id, "root": str(self.root)})

    def _setup_reasoners(self) -> None:
        """Bootstrap the default reasoner roster unless the caller supplied one.

        If an Anthropic API key is available, use the real LLM as the deep
        tier; otherwise fall back to the simulated stub.
        """
        if self.reasoners:
            return
        from .llm import AnthropicReasoner, anthropic_available
        deep = AnthropicReasoner() if anthropic_available() else DeepSimulatedReasoner()
        synth = SynthesisReasoner(library=self.library, tools=self.tools)
        self.reasoners = [FastHeuristicReasoner(), synth, CodeSpecialist(), deep]

    def _setup_router(self) -> None:
        """Build the metacog + router over the assembled reasoner roster."""
        metacog = Metacog(reasoners=self.reasoners)
        self.router = Router(
            reasoners=self.reasoners,
            metacog=metacog,
            min_confidence=self.config.router_min_confidence,
            ms_per_conf=self.config.router_ms_per_conf,
        )

    def _setup_tools_and_subscribers(self) -> None:
        """Register built-in tools and wire state/response persistence hooks."""
        self._register_default_tools()
        self._wire_subscribers()

    def _setup_self_model(self) -> None:
        """Instantiate the SelfModel and wire its hook for fast-tier queries."""
        self.self_model = SelfModel(
            latent=self.latent, slow_mem=self.slow_mem,
            reasoners=self.reasoners, tools=self.tools,
        )
        set_self_model_hook(self.self_model.answer)

    def _setup_background_surfaces(self) -> None:
        """Prepare consolidator + curiosity loops and the curiosity→intent bridge."""
        self.consolidator = Consolidator(
            slow_mem=self.slow_mem,
            latent=self.latent,
            interval_s=self.config.consolidator_interval_s,
        )
        self.curiosity = Curiosity(
            bus=self.bus,
            world=self.world,
            latent=self.latent,
            interval_s=self.config.curiosity_interval_s,
            idle_threshold_s=self.config.curiosity_idle_threshold_s,
        )
        self.curiosity.wire()

        async def _on_curiosity(event: Event) -> None:
            from .types import Intent as _Intent
            text = event.payload.get("text", "")
            if not text:
                return
            await self.handle_intent(_Intent(text=text, source="curiosity"))
        self.bus.subscribe("curiosity.generate", _on_curiosity, name="runtime.curiosity")

    # -- default tool set ---------------------------------------------------
    def _register_default_tools(self) -> None:
        import datetime

        @self.tools.register("clock.now", "Current wall time, ISO-8601.", returns="str", est_cost_ms=0.5)
        async def _clock_now() -> str:
            return datetime.datetime.now().isoformat(timespec="seconds")

        @self.tools.register("memory.note", "Save a free-text note to slow memory.", returns="bool", est_cost_ms=2.0)
        async def _memory_note(fact: str) -> bool:
            self.slow_mem.record("note", "user", {"fact": fact})
            self.semantic_mem.remember(fact, kind="note")
            return True

        @self.tools.register("memory.recall", "Recall all notes.", returns="list[str]", est_cost_ms=3.0)
        async def _memory_recall() -> list[str]:
            rows = self.slow_mem.query(kind="note", limit=25)
            return [r["payload"]["fact"] for r in rows]

        @self.tools.register("memory.search", "Semantic search over notes.", returns="list[dict]", est_cost_ms=5.0)
        async def _memory_search(query: str, k: int = 3) -> list[dict[str, Any]]:
            return self.semantic_mem.search(query, k=k, kind="note")

    # -- subscribers --------------------------------------------------------
    def _wire_subscribers(self) -> None:
        async def log_state(e: Event) -> None:
            self.slow_mem.record("state", e.origin, e.payload, e.trace_id)

        async def log_response(e: Event) -> None:
            self.slow_mem.record("response", e.origin, e.payload, e.trace_id)

        self.bus.subscribe("state.*", log_state, name="persist.state")
        self.bus.subscribe("response.*", log_response, name="persist.response")

    # -- the inference loop -------------------------------------------------
    async def handle_intent(self, intent: Intent) -> Response:
        """Run one turn of the inference pipeline for ``intent``.

        The pipeline is:

        1. **Publish** an ``intent.<source>`` event on the bus so
           subscribers (persistence, debug taps) see the request.
        2. **Route** the intent through :class:`Router`, which picks a
           reasoner (with fall-through on low confidence) and returns the
           chosen :class:`Decision`, the final :class:`Response`, and the
           list of reasoners that were attempted.
        3. **Verify** the response against the intent and world state via
           :class:`Verifier`, yielding an accept/reject verdict with a
           numeric score and reasons.
        4. **Persist** a latent summary of the turn (turn count, last
           intent, last reasoner, last confidence, last verdict score)
           and checkpoint it to disk.
        5. **Publish** a ``response.<tier>`` event carrying the response
           text, tier, tools used, cost, verdict, and the list of
           attempted reasoners — so subscribers (including the default
           ``response.*`` logger) see the outcome.

        Args:
            intent: The user or system intent to handle.

        Returns:
            The :class:`Response` selected by the router (after any
            low-confidence fall-through). The verdict is published on the
            bus but not attached to the return value.
        """
        await self.bus.publish(
            Event(topic=f"intent.{intent.source}", origin=intent.source, trace_id=intent.trace_id,
                  payload={"text": intent.text})
        )
        decision, response, attempted = await self.router.route(intent, self.world, self.tools)
        verdict = self.verifier.verify(intent, response, self.world)

        # Persist latent summary of what happened this turn.
        turn_count = int(self.latent.get("turns", 0)) + 1
        self.latent.update(
            turns=turn_count,
            last_intent=intent.text,
            last_reasoner=decision.reasoner,
            last_confidence=response.confidence,
            last_verdict_score=verdict.score,
        )
        self.latent.checkpoint()

        await self.bus.publish(
            Event(
                topic=f"response.{decision.tier}",
                origin=decision.reasoner,
                trace_id=intent.trace_id,
                confidence=response.confidence,
                payload={
                    "text": response.text,
                    "tier": decision.tier,
                    "tools": response.tools_used,
                    "cost_ms": response.cost_ms,
                    "verdict": {"accept": verdict.accept, "score": verdict.score, "reasons": verdict.reasons},
                    "attempted": [a.reasoner for a in attempted],
                },
            )
        )
        return response

    # -- event loop ---------------------------------------------------------
    async def run(self, inputs: asyncio.Queue[Intent | None]) -> None:
        """Process intents from ``inputs`` until a ``None`` sentinel arrives.

        Args:
            inputs: Queue of intents to handle. Pushing ``None`` cleanly
                terminates the loop.
        """
        while True:
            intent = await inputs.get()
            if intent is None:
                return
            await self.handle_intent(intent)

    async def start(self) -> None:
        """Activate the bus transport (no-op for in-process)."""
        await self.bus.start()

    def start_background(self) -> list[asyncio.Task[None]]:
        """Start the consolidation + curiosity loops as background tasks.

        Resets the internal stop events so the loops can be restarted
        after a previous :meth:`stop_background` call.

        Returns:
            The two created :class:`asyncio.Task` handles, in order:
            consolidator first, curiosity second.
        """
        self._consolidator_stop.clear()
        self._curiosity_stop.clear()
        return [
            asyncio.create_task(self.consolidator.run(self._consolidator_stop)),
            asyncio.create_task(self.curiosity.run(self._curiosity_stop)),
        ]

    def stop_background(self) -> None:
        """Signal the consolidation + curiosity loops to exit.

        Sets the stop events but does not await the tasks — callers that
        need to join the tasks should retain the handles from
        :meth:`start_background`.
        """
        self._consolidator_stop.set()
        self._curiosity_stop.set()

    async def shutdown(self) -> None:
        """Stop loops, run a final consolidation, then close all resources.

        A terminal consolidation ensures the digest reflects the full
        session before the slow-memory connection is torn down. Safe to
        call even if the background loops were never started, and
        idempotent — a second call is a no-op.
        """
        if getattr(self, "_shutdown_done", False):
            return
        self.stop_background()
        # Run one final consolidation so the digest reflects the full session.
        self.consolidator.consolidate()
        self.latent.checkpoint()
        self.slow_mem.close()
        self.semantic_mem.close()
        await self.bus.close()
        self._shutdown_done = True
