"""Synthesizers — pluggable backends that emit Python source from an intent.

A synthesizer maps an intent's natural-language text to a concrete
(source, entrypoint, args) triple. The triple is then validated and executed
by the shared sandbox machinery in ``mente.synthesis``; synthesizers
themselves don't touch the sandbox.

Two implementations ship:
  * ``TemplateSynthesizer`` — deterministic, regex-driven, zero-deps.
  * ``LLMSynthesizer`` — asks Claude for a pure function; returns ``None``
    on refusal / parse failure so the caller can fall back.

The module keeps ``LLMSynthesizer`` behind a lazy import so that environments
without the ``anthropic`` SDK can still import ``TemplateSynthesizer``.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .template import TemplateSynthesizer


@runtime_checkable
class Synthesizer(Protocol):
    """Pluggable synthesis backend.

    ``synthesize`` must return either a ``(source, entrypoint, args)`` triple
    or ``None`` when the backend declines to handle the intent.
    """

    def synthesize(
        self, intent_text: str
    ) -> tuple[str, str, dict[str, Any]] | None: ...


def __getattr__(name: str) -> Any:
    # Lazy-import LLMSynthesizer so that `from mente.synthesizers import
    # TemplateSynthesizer` works when `anthropic` is not installed.
    if name == "LLMSynthesizer":
        from .llm import LLMSynthesizer
        return LLMSynthesizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Synthesizer", "TemplateSynthesizer", "LLMSynthesizer"]
