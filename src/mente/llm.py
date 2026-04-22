"""Anthropic-backed reasoner.

Drop-in replacement for DeepSimulatedReasoner. Activates when ANTHROPIC_API_KEY
is set. Uses Claude Opus 4.7 with adaptive thinking and prompt caching on the
system prompt so turn-over-turn we get a cheap cache read for the frozen
preamble.

Activity guidance (§4 of the architecture — native tool binding): we serialize
the registered tool catalogue and the current world-model snapshot into the
system prompt so the model answers grounded in real state. This is NOT yet
true latent-space tool binding — it's the text-serialized baseline. The real
next step is invoking the Claude tool runner so the model can call MENTE's
tools directly; left as a follow-up.

Operational posture (post-launch hardening):

* reads model / tokens / effort from :class:`MenteConfig` when supplied, so
  the runtime has one source of truth for LLM tuning;
* wraps every API call in a per-instance :class:`CircuitBreaker` (5 failures,
  60s recovery) so a downstream outage stops burning retries;
* emits structured logs (start / retry / exhaustion / circuit-open) tagged
  with ``trace_id`` — never the intent text or model output, which could
  contain secrets or PII.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any

from .config import MenteConfig
from .logging import get_logger
from .resilience import CircuitBreaker, CircuitOpenError, retry_async
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
        anthropic.APIConnectionError,
        anthropic.APIStatusError,
        ConnectionError,
        TimeoutError,
    )
else:  # pragma: no cover
    _LLM_RETRY_ON = (ConnectionError, TimeoutError)


_log = get_logger("llm")


_SYSTEM_PROMPT = """You are MENTE, a persistent, event-driven reasoning process.

You are the 'deep' tier of a heterogeneous cognitive architecture. You are
called when the fast heuristic tier cannot confidently answer. Answer
concisely and precisely — no preamble, no filler.

You have access to the system's current world-model snapshot (ground-truth
beliefs) and tool catalogue below. Ground your answers in that state. If a
fact is not in the world model, say so rather than guessing.

Respond with the final answer only. Do not narrate your reasoning to the
user — your thinking happens separately."""


_CIRCUIT_OPEN_TEXT = "[deep.claude unavailable — circuit open]"


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
    config: MenteConfig | None = None
    # Breaker thresholds are fixed by design; the tunable knobs live on
    # MenteConfig. One breaker per reasoner instance so per-Runtime state
    # stays isolated.
    _breaker: CircuitBreaker = field(
        default_factory=lambda: CircuitBreaker(failure_threshold=5, recovery_s=60.0),
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not _ANTHROPIC_AVAILABLE:
            raise RuntimeError(
                "anthropic SDK not installed. Install with: pip install 'mente[llm]'"
            )
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        # When a config is supplied it wins over the dataclass defaults.
        # We can't distinguish an explicit kwarg from a default, so "config
        # supplied" is the documented contract for "use config values".
        if self.config is not None:
            self.model = self.config.llm_model
            self.max_tokens = self.config.llm_max_tokens
            self.effort = self.config.llm_effort
        # Client is ``Any``-typed so we can pass params (``output_config``,
        # adaptive thinking) that aren't on the installed SDK's typed surface
        # yet, and so the fake test double type-checks cleanly.
        self._client: Any = anthropic.AsyncAnthropic()

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

    def _circuit_open_response(self, intent: Intent, cost_ms: float = 0.0) -> Response:
        _log.warning(
            "llm call rejected: circuit open",
            extra={
                "trace_id": intent.trace_id,
                "reasoner": self.name,
            },
        )
        return Response(
            text=_CIRCUIT_OPEN_TEXT,
            reasoner=self.name,
            tier=self.tier,
            confidence=0.0,
            cost_ms=cost_ms,
        )

    async def answer(
        self, intent: Intent, world: WorldModel, tools: ToolRegistry
    ) -> Response:
        context_block = self._render_context(world, tools)
        # System is a list so we can mark the stable preamble as cacheable.
        # The volatile world-model snapshot comes AFTER the cache breakpoint
        # so preamble reuse survives state changes.
        system: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": context_block},
        ]

        # Fast-path the open-breaker rejection so we skip retry-wrapper setup.
        if self._breaker.state == "open":
            return self._circuit_open_response(intent)

        _log.info(
            "llm call start",
            extra={
                "trace_id": intent.trace_id,
                "model": self.model,
                "effort": self.effort,
                "intent_len": len(intent.text),
            },
        )

        attempts_seen = 0

        async def _one_shot() -> Any:
            nonlocal attempts_seen
            attempts_seen += 1
            if attempts_seen > 1:
                _log.warning(
                    "llm call retry",
                    extra={
                        "trace_id": intent.trace_id,
                        "attempt": attempts_seen,
                    },
                )
            return await self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                system=system,
                messages=[{"role": "user", "content": intent.text}],
            )

        wrapped_call = retry_async(retry_on=_LLM_RETRY_ON)(_one_shot)

        t0 = asyncio.get_running_loop().time()
        try:
            message = await self._breaker.call(wrapped_call)
        except CircuitOpenError:
            return self._circuit_open_response(
                intent, cost_ms=(asyncio.get_running_loop().time() - t0) * 1000
            )
        except Exception as e:
            _log.error(
                "llm call exhausted",
                extra={
                    "trace_id": intent.trace_id,
                    "exc_type": type(e).__name__,
                    "breaker_state": self._breaker.state,
                },
            )
            return Response(
                text=f"[deep.claude error: {type(e).__name__}: {e}]",
                reasoner=self.name,
                tier=self.tier,
                confidence=0.0,
                cost_ms=(asyncio.get_running_loop().time() - t0) * 1000,
            )

        cost_ms = (asyncio.get_running_loop().time() - t0) * 1000
        text = "".join(b.text for b in message.content if b.type == "text").strip()

        # Confidence proxy: end_turn = confident answer; anything else is a signal.
        conf = 0.85 if message.stop_reason == "end_turn" else 0.5

        return Response(
            text=text or "[deep.claude returned empty output]",
            reasoner=self.name,
            tier=self.tier,
            confidence=conf,
            cost_ms=cost_ms,
        )
