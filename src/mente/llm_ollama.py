"""Ollama-backed local reasoner.

Talks to a locally-hosted `Ollama <https://ollama.com>`_ HTTP endpoint so MENTE
can run the ``deep`` tier entirely on-device — no third-party cloud, no API key,
no per-token cost. Pair with :class:`mente.llm.AnthropicReasoner` via Unit 3's
auto-selector so the runtime falls back to Ollama when ``ANTHROPIC_API_KEY`` is
absent but a local model is reachable.

Phase 1 (this module): text-serialized world-model snapshot + tool catalogue
injected into a Chat-API system prompt, single non-streaming POST to
``/api/chat``, naive confidence (``0.75`` on a well-formed reply, ``0.0`` on any
failure). No retries, no circuit breaker — the local endpoint is under the
user's control, so a failure is signal to the router, not an incident.

Phase 2: streaming decode, native tool-call protocol (Ollama added ``tools``
support in 0.1.44), structured-output JSON schemas, and parity with the
resilience stack in :mod:`mente.llm` (retries + breaker) so remote Ollama
deployments behave like first-class reasoners.

Operational posture:

* ``httpx`` is a hard dep of this module but an *optional* dep of the project;
  users opt in via ``pip install 'mente[llm-ollama]'``. The import is lazy so
  merely importing :mod:`mente` does not pull ``httpx`` into memory.
* never logs ``intent.text`` or the model's reply at INFO/WARNING — local models
  still see PII, and a future shared log drain should not leak it.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from .logging import get_logger
from .tools import ToolRegistry
from .types import Intent, ReasonerTier, Response
from .world_model import WorldModel

_log = get_logger("llm_ollama")


_IMPORT_ERROR_MSG = (
    "httpx not installed. Install with: pip install 'mente[llm-ollama]'"
)


_SYSTEM_PROMPT = """You are MENTE, a persistent, event-driven reasoning process.

You are the 'deep' tier of a heterogeneous cognitive architecture running
locally via Ollama. You are called when the fast heuristic tier cannot
confidently answer. Answer concisely and precisely — no preamble, no filler.

You have access to the system's current world-model snapshot (ground-truth
beliefs) and tool catalogue below. Ground your answers in that state. If a
fact is not in the world model, say so rather than guessing.

Respond with the final answer only. Do not narrate your reasoning to the user."""


def _load_httpx() -> Any:
    """Import ``httpx`` lazily with a friendly error if it is missing.

    Returns:
        The imported ``httpx`` module.

    Raises:
        ImportError: When ``httpx`` is not installed; the message points
            users at the ``llm-ollama`` optional-dep extra.
    """
    try:
        import httpx  # noqa: PLC0415 — lazy by design
    except ImportError as exc:  # pragma: no cover - defensive
        raise ImportError(_IMPORT_ERROR_MSG) from exc
    return httpx


def ollama_available(url: str = "http://127.0.0.1:11434") -> bool:
    """Return ``True`` when the Ollama server at ``url`` answers ``/api/tags``.

    Used by the runtime's auto-selector to decide whether to wire up an
    :class:`OllamaReasoner`. Swallows every exception — this is a probe, not
    a health check — and caps the wait at one second so startup stays snappy.

    Args:
        url: Base URL of the Ollama HTTP endpoint (no trailing path).

    Returns:
        ``True`` on HTTP 200, ``False`` for any network error, non-200, or
        missing ``httpx`` dependency.
    """
    try:
        httpx = _load_httpx()
    except ImportError:
        return False
    try:
        resp = httpx.get(f"{url.rstrip('/')}/api/tags", timeout=1.0)
    except Exception:
        return False
    return resp.status_code == 200


@dataclass
class OllamaReasoner:
    """Deep-tier reasoner that calls a local Ollama Chat API.

    Speaks the Ollama ``POST /api/chat`` protocol with ``stream=false``. The
    request carries a grounded system prompt (world-model + tool catalogue)
    and a single user message (``intent.text``); the response is the parsed
    ``message.content`` string.

    Attributes:
        name: Router-visible identifier (``"deep.ollama"``).
        tier: Reasoner tier; always ``"deep"``.
        est_cost_ms: Rough p50 latency estimate used by the router.
        url: Base URL of the Ollama server.
        model: Ollama model tag (e.g. ``"llama3.2"``, ``"qwen2.5:14b"``).
        max_tokens: Upper bound on generated tokens (``num_predict``).
        timeout_s: Per-request HTTP timeout in seconds.
    """

    name: str = "deep.ollama"
    tier: ReasonerTier = "deep"
    est_cost_ms: float = 1500.0
    url: str = "http://127.0.0.1:11434"
    model: str = "llama3.2"
    max_tokens: int = 1024
    timeout_s: float = 30.0
    _client: Any = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        """Eagerly resolve ``httpx`` so missing deps fail fast at construction.

        Raises:
            ImportError: When ``httpx`` is not installed.
        """
        self._client = _load_httpx()

    def _render_context(self, world: WorldModel, tools: ToolRegistry) -> str:
        """Serialize the world-model snapshot + tool catalogue into text.

        Mirrors the layout used by :class:`mente.llm.AnthropicReasoner` so a
        user switching between backends sees identical grounding context.

        Args:
            world: Current world-model snapshot.
            tools: Typed tool registry (catalogue only; not invoked here).

        Returns:
            A plain-text block ready to concatenate into the system prompt.
        """
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

    def _error_response(self, message: str, cost_ms: float) -> Response:
        """Build the canonical zero-confidence error response.

        The text prefix matches the pattern in :mod:`mente.llm` so downstream
        log scrapers / UIs can bucket deep-tier failures uniformly.
        """
        return Response(
            text=f"[deep.ollama error: {message}]",
            reasoner=self.name,
            tier=self.tier,
            confidence=0.0,
            cost_ms=cost_ms,
        )

    async def answer(
        self, intent: Intent, world: WorldModel, tools: ToolRegistry
    ) -> Response:
        """Send ``intent`` to Ollama and return a grounded :class:`Response`.

        On success the response carries ``confidence=0.75``. Every failure path
        (transport error, non-200, JSON decode, missing ``message.content``)
        collapses into a single zero-confidence error response so the router
        can escalate or fall back without inspecting exception types.

        Args:
            intent: The incoming user/system intent.
            world: Current world-model snapshot; rendered into the prompt.
            tools: Tool registry; catalogue rendered into the prompt.

        Returns:
            A :class:`Response` with ``confidence=0.75`` on success or
            ``confidence=0.0`` with an ``[deep.ollama error: ...]`` body on
            any failure.
        """
        context_block = self._render_context(world, tools)
        system_prompt = f"{_SYSTEM_PROMPT}\n\n{context_block}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": intent.text},
            ],
            "stream": False,
            "options": {"num_predict": self.max_tokens},
        }
        endpoint = f"{self.url.rstrip('/')}/api/chat"

        _log.info(
            "ollama call start",
            extra={
                "trace_id": intent.trace_id,
                "model": self.model,
                "intent_len": len(intent.text),
            },
        )

        t0 = asyncio.get_running_loop().time()
        try:
            async with self._client.AsyncClient(timeout=self.timeout_s) as client:
                resp = await client.post(endpoint, json=payload)
        except Exception as e:
            cost_ms = (asyncio.get_running_loop().time() - t0) * 1000
            _log.error(
                "ollama call failed",
                extra={
                    "trace_id": intent.trace_id,
                    "exc_type": type(e).__name__,
                },
            )
            return self._error_response(f"{type(e).__name__}: {e}", cost_ms)

        cost_ms = (asyncio.get_running_loop().time() - t0) * 1000

        status = resp.status_code
        if status != 200:
            _log.error(
                "ollama non-200",
                extra={
                    "trace_id": intent.trace_id,
                    "status": status,
                },
            )
            return self._error_response(f"HTTP {status}", cost_ms)

        try:
            data = resp.json()
        except Exception as e:
            _log.error(
                "ollama malformed json",
                extra={
                    "trace_id": intent.trace_id,
                    "exc_type": type(e).__name__,
                },
            )
            return self._error_response(f"malformed JSON: {type(e).__name__}", cost_ms)

        message = data.get("message") if isinstance(data, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            _log.error(
                "ollama missing content",
                extra={"trace_id": intent.trace_id},
            )
            return self._error_response("missing message.content", cost_ms)

        return Response(
            text=content.strip(),
            reasoner=self.name,
            tier=self.tier,
            confidence=0.75,
            cost_ms=cost_ms,
        )
