"""LLM-authored synthesizer — asks Claude for a pure Python function.

The synthesizer is deliberately conservative: it demands a JSON object with
``source``, ``entrypoint``, and ``args``; strips any markdown fences; and
returns ``None`` on refusal or parse failure so the caller can fall back to
the template synthesizer. Produced source is NOT trusted — the shared
``aria.synthesis._validate_ast`` / ``_run_sandboxed`` gates in the
``SynthesisReasoner`` still police the output.

This synthesizer has its own Anthropic client; it does not depend on, and is
not coupled to, ``aria.llm``.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:  # pragma: no cover
    anthropic = None  # type: ignore[assignment]
    _ANTHROPIC_AVAILABLE = False


_SYSTEM_PROMPT = """You are a program-synthesis assistant for ARIA.

Given a natural-language intent describing a computation, emit a SINGLE pure
Python function that computes it, plus the call arguments extracted from the
intent.

HARD CONSTRAINTS on the function source:
  * No import statements.
  * No I/O (no open, input, print, no file access, no network).
  * No exec / eval / compile / __import__.
  * No global, nonlocal, try/except/raise, with, async, lambda, or attribute
    access to dunders (anything __like_this__).
  * Pure computation only — deterministic, terminating, no side effects.
  * The function must be callable with the JSON "args" dict as **kwargs.

OUTPUT FORMAT — respond with a SINGLE JSON object and nothing else:
{
  "source": "def foo(n):\\n    ...\\n",
  "entrypoint": "foo",
  "args": {"n": 10}
}

If the request is NOT a pure-computation intent, or you cannot satisfy the
constraints, respond with exactly: {"refuse": true}
"""


# Matches ```json ... ``` or ``` ... ``` fences around a JSON blob.
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.match(text)
    return m.group(1) if m else text.strip()


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort JSON parse: strip fences, then try the whole string, then
    fall back to the first {...} balanced substring."""
    text = _strip_fences(text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # Balanced-brace scan; handles a chatty prefix/suffix from the model.
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        return obj if isinstance(obj, dict) else None
                    except Exception:
                        break
        start = text.find("{", start + 1)
    return None


@dataclass
class LLMSynthesizer:
    """Synthesizer that asks Claude to author a pure function.

    The call is async under the hood; ``synthesize`` is sync to match the
    ``Synthesizer`` protocol. Use ``asynthesize`` from async contexts to avoid
    spawning a temporary event loop.
    """

    api_key: str | None = None
    model: str = "claude-opus-4-7"
    max_tokens: int = 2048
    _client: Any = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if not _ANTHROPIC_AVAILABLE:
            raise RuntimeError(
                "anthropic SDK not installed. Install with: pip install 'aria[llm]'"
            )
        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.AsyncAnthropic(api_key=key)

    # --- public API --------------------------------------------------------

    async def asynthesize(
        self, intent_text: str
    ) -> tuple[str, str, dict[str, Any]] | None:
        """Async entrypoint. Returns (source, entrypoint, args) or None."""
        system = [
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            },
        ]
        try:
            message = await self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                thinking={"type": "adaptive"},
                output_config={"effort": "medium"},
                system=system,
                messages=[{"role": "user", "content": intent_text}],
            )
        except Exception:
            return None

        text = "".join(
            getattr(b, "text", "") for b in message.content
            if getattr(b, "type", None) == "text"
        ).strip()
        if not text:
            return None

        obj = _extract_json_object(text)
        if obj is None or obj.get("refuse"):
            return None

        source = obj.get("source")
        entrypoint = obj.get("entrypoint")
        args = obj.get("args", {})
        if not isinstance(source, str) or not isinstance(entrypoint, str):
            return None
        if not isinstance(args, dict):
            return None
        return source, entrypoint, args

    def synthesize(
        self, intent_text: str
    ) -> tuple[str, str, dict[str, Any]] | None:
        """Sync Protocol-compatible entrypoint.

        If called from within a running event loop, raises — use
        ``asynthesize`` from async code paths.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No loop running — safe to spin one up.
            return asyncio.run(self.asynthesize(intent_text))
        raise RuntimeError(
            "LLMSynthesizer.synthesize() called from a running event loop; "
            "use `await LLMSynthesizer.asynthesize(...)` instead."
        )
