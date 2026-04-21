"""Templated synthesis for a small family of computation shapes.

Regex-driven; no LLM involved. Recognizes "fibonacci of N", "factorial of N",
"A to the power of B" (and a handful of phrasings), and emits a pure Python
function plus call args. Moved verbatim from ``aria.synthesis`` so the main
module can expose it as a re-export shim.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


_FIB_RE = re.compile(r"(?:fib(?:onacci)?\D+(\d+)|(\d+)\D+fib(?:onacci)?)", re.I)
_POW_RE = re.compile(r"(\d+)\s*(?:\*\*|\^|to the power of|to the)\s*(\d+)", re.I)
_FACT_RE = re.compile(r"factorial(?:\s+of)?\s+(\d+)|(\d+)\D+factorial", re.I)


@dataclass
class TemplateSynthesizer:
    """Recognizes a small family of computation requests and emits Python
    source to compute them. No LLM involved — replace with an LLM call in
    Phase 2."""

    def synthesize(self, intent_text: str) -> tuple[str, str, dict] | None:
        """Return (source, entrypoint, args) or None if we can't synthesize."""
        m = _FIB_RE.search(intent_text)
        if m:
            n = int(m.group(1) or m.group(2))
            src = (
                "def fib(n):\n"
                "    a, b = 0, 1\n"
                "    for _ in range(n):\n"
                "        a, b = b, a + b\n"
                "    return a\n"
            )
            return src, "fib", {"n": n}
        m = _FACT_RE.search(intent_text)
        if m:
            n = int(m.group(1) or m.group(2))
            src = (
                "def factorial(n):\n"
                "    out = 1\n"
                "    for k in range(2, n + 1):\n"
                "        out *= k\n"
                "    return out\n"
            )
            return src, "factorial", {"n": n}
        m = _POW_RE.search(intent_text)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            src = "def power(a, b):\n    return a ** b\n"
            return src, "power", {"a": a, "b": b}
        return None
