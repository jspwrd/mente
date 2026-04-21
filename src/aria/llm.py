"""Anthropic-backed reasoner.

Drop-in replacement for DeepSimulatedReasoner. Activates when ANTHROPIC_API_KEY
is set. Uses Claude Opus 4.7 with adaptive thinking and prompt caching on the
system prompt so turn-over-turn we get a cheap cache read for the frozen
preamble.

Activity guidance (§4 of the architecture — native tool binding): we serialize
the registered tool catalogue and the current world-model snapshot into the
system prompt so the model answers grounded in real state. This is NOT yet
true latent-space tool binding — it's the text-serialized baseline. The real
next step is invoking the Claude tool runner so the model can call ARIA's
tools directly; left as a follow-up.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

from .resilience import retry_async
from .tools import ToolRegistry
from .types import Intent, ReasonerTier, Response
from .world_model import WorldModel

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:  # pragma: no cover
    anthropic = None  # type: ignore[assignment]
    _ANTHROPIC_AVAILABLE = False

if _ANTHROPIC_AVAILABLE:
    _LLM_RETRY_ON: tuple[type[BaseException], ...] = (
        anthropic.APIConnectionError,  # type: ignore[attr-defined]
        anthropic.APIStatusError,  # type: ignore[attr-defined]
        ConnectionError,
        TimeoutError,
    )
else:  # pragma: no cover
    _LLM_RETRY_ON = (ConnectionError, TimeoutError)


_SYSTEM_PROMPT = """You are ARIA, a persistent, event-driven reasoning process.

You are the 'deep' tier of a heterogeneous cognitive architecture. You are
called when the fast heuristic tier cannot confidently answer. Answer
concisely and precisely — no preamble, no filler.

You have access to the system's current world-model snapshot (ground-truth
beliefs) and tool catalogue below. Ground your answers in that state. If a
fact is not in the world model, say so rather than guessing.

Respond with the final answer only. Do not narrate your reasoning to the
user — your thinking happens separately."""


def anthropic_available() -> bool:
    return _ANTHROPIC_AVAILABLE and bool(os.environ.get("ANTHROPIC_API_KEY"))


@dataclass
class AnthropicReasoner:
    name: str = "deep.claude"
    tier: ReasonerTier = "deep"
    est_cost_ms: float = 3500.0
    model: str = "claude-opus-4-7"
    max_tokens: int = 4096
    effort: str = "medium"

    def __post_init__(self) -> None:
        if not _ANTHROPIC_AVAILABLE:
            raise RuntimeError(
                "anthropic SDK not installed. Install with: pip install 'aria[llm]'"
            )
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.AsyncAnthropic()

    def _render_context(self, world: WorldModel, tools: ToolRegistry) -> str:
        lines: list[str] = []
        lines.append("WORLD MODEL:")
        snapshot = sorted(world.entities())
        if not snapshot:
            lines.append("  (no beliefs yet)")
        else:
            for ent in snapshot:
                for attr, val in world.entity(ent).items():
                    lines.append(f"  {ent}.{attr} = {val!r}")
        lines.append("")
        lines.append("AVAILABLE TOOLS (invoked by the harness, not you — informational):")
        for t in tools.list():
            params = ", ".join(f"{k}: {v}" for k, v in t.params.items())
            lines.append(f"  {t.name}({params}) -> {t.returns}  // {t.description}")
        return "\n".join(lines)

    async def answer(
        self, intent: Intent, world: WorldModel, tools: ToolRegistry
    ) -> Response:
        context_block = self._render_context(world, tools)
        # System is a list so we can mark the stable preamble as cacheable.
        # The volatile world-model snapshot comes AFTER the cache breakpoint
        # so preamble reuse survives state changes.
        system = [
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": context_block},
        ]

        t0 = asyncio.get_event_loop().time()
        try:
            message = await retry_async(retry_on=_LLM_RETRY_ON)(
                self._client.messages.create
            )(
                model=self.model,
                max_tokens=self.max_tokens,
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                system=system,
                messages=[{"role": "user", "content": intent.text}],
            )
        except Exception as e:
            return Response(
                text=f"[deep.claude error: {type(e).__name__}: {e}]",
                reasoner=self.name, tier=self.tier,
                confidence=0.0, cost_ms=(asyncio.get_event_loop().time() - t0) * 1000,
            )

        cost_ms = (asyncio.get_event_loop().time() - t0) * 1000
        text = "".join(b.text for b in message.content if b.type == "text").strip()

        # Confidence proxy: end_turn = confident answer; anything else is a signal.
        conf = 0.85 if message.stop_reason == "end_turn" else 0.5

        return Response(
            text=text or "[deep.claude returned empty output]",
            reasoner=self.name, tier=self.tier,
            confidence=conf, cost_ms=cost_ms,
        )
