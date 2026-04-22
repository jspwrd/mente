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

Drop-in implementation example::

    class PassthroughSynthesizer:
        def synthesize(self, intent_text: str):
            if "double" not in intent_text:
                return None
            src = "def run(x):\\n    return x * 2\\n"
            return src, "run", {"x": 21}
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .template import TemplateSynthesizer


@runtime_checkable
class Synthesizer(Protocol):
    """Pluggable synthesis backend: intent text → executable source.

    mente calls a Synthesizer from ``mente.synthesis`` when an intent is
    routed to the synthesis-execution path. The returned triple is then
    AST-validated and executed inside the sandbox; synthesizers MUST NOT
    execute code themselves.

    Concurrency: ``synthesize`` is synchronous and called from mente's
    event loop. Backends that perform blocking I/O (e.g. remote LLM calls)
    should keep latency bounded or offer an async variant outside this
    Protocol. A Synthesizer SHOULD:

      * return ``None`` on refusal, parse failure, or any non-fatal
        decline, so the caller can fall back to another backend.
      * emit only pure functions with a stable ``entrypoint`` name.
      * populate ``args`` with JSON-serializable values only — the sandbox
        relies on this for validation and tracing.
    """

    def synthesize(
        self, intent_text: str
    ) -> tuple[str, str, dict[str, Any]] | None:
        """Turn natural-language intent text into an executable program.

        Args:
            intent_text: The intent's raw text, exactly as the user or
                upstream component provided it. Not pre-processed.

        Returns:
            Either ``None`` — meaning "I decline to handle this, please
            fall back" — or a triple ``(source, entrypoint, args)`` where:

              * ``source`` is a Python module source string defining a
                single pure function.
              * ``entrypoint`` is the name of that function inside
                ``source``.
              * ``args`` is a ``dict[str, Any]`` of JSON-serializable
                keyword arguments to pass to the entrypoint.

        Raises:
            Exception: Only for genuine backend failures (e.g. network
                errors in LLM-backed synthesizers). Routine decline MUST
                be signalled with ``None`` so the pipeline can fall back.
        """
        ...


def __getattr__(name: str) -> Any:
    # Lazy-import LLMSynthesizer so that `from mente.synthesizers import
    # TemplateSynthesizer` works when `anthropic` is not installed.
    if name == "LLMSynthesizer":
        from .llm import LLMSynthesizer
        return LLMSynthesizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Synthesizer", "TemplateSynthesizer", "LLMSynthesizer"]
