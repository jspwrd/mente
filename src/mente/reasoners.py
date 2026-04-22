"""Reasoner interface + stub implementations.

A Reasoner takes an Intent and the current world-model snapshot and produces
a Response. Reasoners are the swap point for real LLMs (Claude, local Llama,
specialist models).

Phase 1 ships two stubs:
  - FastHeuristicReasoner: pattern-matches simple intents instantly. Cheap,
    confident only on known shapes.
  - DeepSimulatedReasoner: simulates a heavyweight model via artificial
    latency and slightly fuzzier pattern-matching. Stand-in for a frontier LLM.

Phase 2: add AnthropicReasoner (Claude API, prompt caching), LocalReasoner
(llama.cpp), and specialist reasoners (code, math, retrieval).

Drop-in implementation example::

    from dataclasses import dataclass
    from mente.types import Intent, ReasonerTier, Response
    from mente.tools import ToolRegistry
    from mente.world_model import WorldModel

    @dataclass
    class EchoReasoner:
        name: str = "echo"
        tier: ReasonerTier = "fast"
        est_cost_ms: float = 1.0

        async def answer(self, intent: Intent, world: WorldModel, tools: ToolRegistry) -> Response:
            return Response(text=intent.text, reasoner=self.name, tier=self.tier,
                            confidence=0.5, cost_ms=self.est_cost_ms)
"""
from __future__ import annotations

import asyncio
import random
import re
from dataclasses import dataclass
from typing import Protocol

from .tools import ToolRegistry
from .types import Intent, ReasonerTier, Response
from .world_model import WorldModel

# Forward-ref for the self-model dependency — injected at runtime to keep
# the reasoner interface clean.
_SELF_MODEL_HOOK = {"fn": None}


def set_self_model_hook(fn) -> None:  # type: ignore[no-untyped-def]
    """Allow runtime to inject a self-model query function."""
    _SELF_MODEL_HOOK["fn"] = fn


class Reasoner(Protocol):
    """A pluggable reasoner tier that produces a Response for an Intent.

    Reasoners are dispatched by the Router based on predicted confidence and
    cost (see ``mente.metacog.Metacog.estimate``). A Reasoner SHOULD:

      * return a ``Response`` with ``confidence=0.0`` when it does not
        recognize the intent — the Router then escalates to the next tier.
      * never raise for a routine "couldn't answer"; surface that as low
        confidence instead so the pipeline stays observable.
      * be async-safe: mente calls ``answer`` from an event-loop task and may
        run reasoners concurrently.

    Attributes:
        name: Short stable identifier used in logs, verdicts, and traces.
            Convention: ``"<tier>.<impl>"`` (e.g. ``"fast.heuristic"``).
        tier: One of ``"fast"``, ``"specialist"``, ``"deep"``. Controls
            dispatch priority and escalation ordering.
        est_cost_ms: Rough p50 latency estimate in milliseconds. The Router
            uses this to trade off cost vs. confidence — not a hard bound.
    """

    name: str
    tier: ReasonerTier
    est_cost_ms: float

    async def answer(
        self, intent: Intent, world: WorldModel, tools: ToolRegistry
    ) -> Response:
        """Produce a Response for the given Intent.

        Args:
            intent: The parsed user/system intent to handle. ``intent.text``
                holds the natural-language payload.
            world: Read-only snapshot of the current world model. Reasoners
                may query it for context but should not mutate it directly.
            tools: Typed tool registry. Reasoners may invoke tools during
                reasoning (``await tools.invoke("clock.now")``).

        Returns:
            A ``Response`` carrying the answer text, the emitting reasoner's
            ``name``/``tier``, a ``confidence`` in ``[0.0, 1.0]``, an actual
            ``cost_ms``, and any ``tools_used``. Set ``confidence=0.0`` to
            signal "don't use mine" so the Router escalates.

        Raises:
            Exception: Only for genuine infrastructure failures (network,
                tool-registry misuse, etc.). Routine "I don't know" is
                expressed via ``confidence=0.0``, not exceptions.
        """
        ...


# --- Stubs ------------------------------------------------------------------

_SIMPLE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^\s*(hi|hello|hey)\b", re.I), "Hi — I'm online."),
    (re.compile(r"\bwhat time is it\b", re.I), "time:now"),  # sentinel → tool
    (re.compile(r"\bwho am i\b", re.I), "user:lookup"),
    (re.compile(r"\bremember that (.+)$", re.I), "remember:capture"),
    (re.compile(r"\bwhat do you remember\b", re.I), "remember:recall"),
    (re.compile(r"\b(what are you|what can you do|how many turns|describe yourself|your reasoners|your tools|what have you been doing)\b", re.I), "self:query"),
    (re.compile(r"\bwhat do you know about (.+)$", re.I), "semantic:search"),
]


@dataclass
class FastHeuristicReasoner:
    """Cheap pattern-matching reasoner for known intent shapes.

    Scans the intent text against a small table of regexes. On a hit it
    either returns a canned reply, dispatches a tool (``clock.now``,
    ``memory.note``, ``memory.recall``, ``memory.search``), or defers to
    the self-model via the injected hook. On a miss it returns an empty
    response with ``confidence=0``, signalling the router to escalate.

    Covered patterns:
        - Greetings (``hi``/``hello``/``hey``) → fixed reply.
        - ``what time is it`` → ``clock.now`` tool.
        - ``who am i`` → lookup via the world model.
        - ``remember that <fact>`` → ``memory.note`` tool.
        - ``what do you remember`` → ``memory.recall`` tool.
        - ``what do you know about <topic>`` → ``memory.search`` tool.
        - Self-referential queries (``what are you``, ``how many turns``,
          ``describe yourself``, etc.) → self-model hook if attached.

    Attributes:
        name: Reasoner identifier used by the router.
        tier: Router tier label; always ``"fast"``.
        est_cost_ms: Predicted latency used by metacog estimates.
    """
    name: str = "fast.heuristic"
    tier: ReasonerTier = "fast"
    est_cost_ms: float = 2.0

    async def answer(
        self, intent: Intent, world: WorldModel, tools: ToolRegistry
    ) -> Response:
        """Try each pattern in order; return the first matching reply.

        Self-referential patterns dispatch to the module-level self-model
        hook (set via :func:`set_self_model_hook`); if no hook is attached
        the reply falls back to a diagnostic string but still reports high
        confidence so the router does not escalate needlessly.

        Args:
            intent: The incoming user intent.
            world: Current world-model snapshot (used for ``who am i``).
            tools: Tool registry for ``clock.now``, ``memory.*``.

        Returns:
            A :class:`Response` with ``confidence > 0`` when a pattern
            matched, or an empty response with ``confidence=0`` when none
            did (so the router escalates to a deeper tier).
        """
        tools_used: list[str] = []
        text = intent.text.strip()
        for pat, action in _SIMPLE_PATTERNS:
            m = pat.search(text)
            if not m:
                continue
            if action == "time:now":
                r = await tools.invoke("clock.now")
                tools_used.append("clock.now")
                return Response(
                    text=f"It is {r.value}.",
                    reasoner=self.name, tier=self.tier,
                    confidence=0.98, cost_ms=r.cost_ms, tools_used=tools_used,
                )
            if action == "user:lookup":
                name = world.get("user", "name")
                v = name.value if name else "someone I don't know yet"
                return Response(
                    text=f"You are {v}.",
                    reasoner=self.name, tier=self.tier,
                    confidence=0.9 if name else 0.4, cost_ms=self.est_cost_ms,
                )
            if action == "remember:capture":
                fact = m.group(1).rstrip(".!?")
                r = await tools.invoke("memory.note", fact=fact)
                tools_used.append("memory.note")
                return Response(
                    text=f"Noted: {fact}.",
                    reasoner=self.name, tier=self.tier,
                    confidence=0.95, cost_ms=r.cost_ms, tools_used=tools_used,
                )
            if action == "semantic:search":
                topic = m.group(1).rstrip(".!?")
                r = await tools.invoke("memory.search", query=topic, k=3)
                tools_used.append("memory.search")
                hits = r.value or []
                if not hits:
                    body = "nothing relevant"
                else:
                    body = "; ".join(f"{h['text']} (score {h['score']:.2f})" for h in hits)
                return Response(
                    text=f"About '{topic}': {body}.",
                    reasoner=self.name, tier=self.tier,
                    confidence=0.85 if hits else 0.4,
                    cost_ms=r.cost_ms, tools_used=tools_used,
                )
            if action == "self:query":
                hook = _SELF_MODEL_HOOK["fn"]
                reply = hook(intent.text) if hook else "self-model not attached"
                return Response(
                    text=reply,
                    reasoner=self.name, tier=self.tier,
                    confidence=0.92, cost_ms=self.est_cost_ms,
                )
            if action == "remember:recall":
                r = await tools.invoke("memory.recall")
                tools_used.append("memory.recall")
                notes = r.value or []
                body = "; ".join(notes) if notes else "nothing yet"
                return Response(
                    text=f"I remember: {body}.",
                    reasoner=self.name, tier=self.tier,
                    confidence=0.9, cost_ms=r.cost_ms, tools_used=tools_used,
                )
            return Response(
                text=action,
                reasoner=self.name, tier=self.tier,
                confidence=0.9, cost_ms=self.est_cost_ms,
            )
        # No pattern matched — low confidence, router should escalate.
        return Response(
            text="",
            reasoner=self.name, tier=self.tier,
            confidence=0.0, cost_ms=self.est_cost_ms,
        )


_DEEP_INSTALL_HINT = (
    "(deep-sim fallback; install `mente[llm-ollama]` or set "
    "`ANTHROPIC_API_KEY` for a real LLM)"
)

# "what do you know about X" — we use this to decide whether to reach for
# memory.search. Mirrors the shape FastHeuristicReasoner matches, but we only
# get here when Fast declined (or the router jumped straight to deep).
_DEEP_KNOW_ABOUT = re.compile(r"\bwhat do you know about (.+)$", re.I)
_DEEP_TIME_HINT = re.compile(r"\b(when|what time|current time|today|now)\b", re.I)


@dataclass
class DeepSimulatedReasoner:
    """Stand-in for a heavyweight LLM; composes a useful offline fallback.

    Replace with ``AnthropicReasoner`` or ``LocalReasoner`` in Phase 2
    without touching anything that depends on the Reasoner protocol. Useful
    for tests and demos that need a deep-tier response shape without
    hitting a real model.

    The fallback is intentionally more than a canned echo: it surfaces
    matching world-model entities, runs ``memory.search`` for
    "what do you know about X" shapes, and invokes relevant tools (e.g.
    ``clock.now`` on temporal intents). Every reply ends with an install
    hint so users know they're hitting the stub, not a real LLM.

    Attributes:
        name: Reasoner identifier used by the router.
        tier: Router tier label; always ``"deep"``.
        est_cost_ms: Simulated latency used by metacog estimates and to
            calibrate the artificial sleep in :meth:`answer`.
    """
    name: str = "deep.sim"
    tier: ReasonerTier = "deep"
    est_cost_ms: float = 400.0

    async def answer(
        self, intent: Intent, world: WorldModel, tools: ToolRegistry
    ) -> Response:
        """Compose a thoughtful offline reply from world model + tools.

        The fallback assembles up to three evidence fragments, in priority
        order: matching world-model entities, a ``memory.search`` top hit
        (for "what do you know about X" shapes), and a ``clock.now``
        reading when the intent smells temporal. If nothing useful
        surfaces, the reply degrades to a brief acknowledgement. The
        install hint is always appended so the caller knows this is a
        stub, not a real LLM.

        Confidence stays at ``0.55`` and ``est_cost_ms`` at ``400`` so
        router escalation semantics are unchanged. The artificial sleep is
        preserved so metrics match a real heavyweight call.

        Args:
            intent: The incoming user intent.
            world: Current world-model snapshot; scanned for entity names
                that appear in ``intent.text`` (substring match).
            tools: Tool registry; ``memory.search`` and ``clock.now`` are
                invoked when relevant. Malformed tool outputs degrade
                silently.

        Returns:
            A :class:`Response` whose text ends with the install hint and
            whose ``confidence`` is a steady ``0.55``.
        """
        # Simulate a forward pass + thinking budget.
        await asyncio.sleep(self.est_cost_ms / 1000.0 * random.uniform(0.8, 1.2))

        text = intent.text.strip()
        text_lc = text.lower()
        fragments: list[str] = []
        tools_used: list[str] = []

        # 1. World-model entity match (fuzzy substring).
        entity_hit = self._match_entity(world, text_lc)
        if entity_hit is not None:
            ent_name, attrs = entity_hit
            if attrs:
                attr_str = ", ".join(f"{k}={v}" for k, v in attrs.items())
                fragments.append(
                    f"From the world model, I know '{ent_name}' ({attr_str})."
                )
            else:
                fragments.append(f"I've seen the entity '{ent_name}' before.")

        # 2. Semantic memory search for "what do you know about X".
        know_match = _DEEP_KNOW_ABOUT.search(text)
        if know_match and tools.get("memory.search") is not None:
            topic = know_match.group(1).rstrip(".!?")
            top = await self._top_memory_hit(tools, topic)
            if top is not None:
                tools_used.append("memory.search")
                fragments.append(
                    f"Memory recalls: '{top['text']}' (score {top['score']:.2f})."
                )

        # 3. Clock tool for temporal intents.
        if _DEEP_TIME_HINT.search(text) and tools.get("clock.now") is not None:
            clock_value = await self._safe_clock(tools)
            if clock_value is not None:
                tools_used.append("clock.now")
                fragments.append(f"The current time is {clock_value}.")

        if fragments:
            body = " ".join(fragments)
            reply = f"[deep] Thinking about '{text}': {body} {_DEEP_INSTALL_HINT}"
        else:
            reply = (
                f"[deep] I considered '{text}' but couldn't ground it in "
                f"world-model state or memory. {_DEEP_INSTALL_HINT}"
            )

        return Response(
            text=reply,
            reasoner=self.name, tier=self.tier,
            confidence=0.55, cost_ms=self.est_cost_ms,
            tools_used=tools_used,
        )

    @staticmethod
    def _match_entity(
        world: WorldModel, text_lc: str
    ) -> tuple[str, dict[str, object]] | None:
        """Return ``(name, attrs)`` for the first world entity mentioned in ``text_lc``.

        Matches case-insensitive substring. Returns ``None`` if the world is
        empty or no entity name appears in the text.
        """
        for name in world.entities():
            if name and name.lower() in text_lc:
                return name, world.entity(name)
        return None

    @staticmethod
    async def _top_memory_hit(
        tools: ToolRegistry, topic: str
    ) -> dict[str, object] | None:
        """Invoke ``memory.search`` and return the top hit, or ``None`` on miss.

        Swallows malformed tool outputs (missing keys, non-list values) so
        the fallback never raises for routine degradation.
        """
        result = await tools.invoke("memory.search", query=topic)
        if not result.ok:
            return None
        try:
            hits = list(result.value or [])
        except TypeError:
            return None
        for h in hits:
            if isinstance(h, dict) and "text" in h and "score" in h:
                return h
        return None

    @staticmethod
    async def _safe_clock(tools: ToolRegistry) -> object | None:
        """Invoke ``clock.now`` and return the value, or ``None`` on failure."""
        result = await tools.invoke("clock.now")
        return result.value if result.ok else None
