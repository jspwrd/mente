"""Coding agent example: synthesis + library reuse.

Demonstrates one of MENTE's distinctive capabilities: when an intent has a
"computation shape" the synthesis reasoner writes a small Python function,
validates it (AST gate + sandbox execution), and on success *promotes* the
function into a persistent library. The next time a same-shape intent
arrives, the library primitive is reused — its `invocations` counter
increments instead of re-synthesizing.

Run:
    python examples/coding_agent.py
"""
from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

# Make the in-tree `mente` package importable without `pip install`.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from mente.runtime import Runtime  # noqa: E402
from mente.types import Intent  # noqa: E402


# Each example uses a dedicated data dir so its state doesn't collide with
# the main REPL's `.mente/` or other examples.
DATA_DIR = _ROOT / ".mente-example-coding"


# Scripted conversation. Each "round" hits a different computation shape; the
# second round repeats the *shapes* so we can observe library reuse.
SCRIPT_ROUND_1 = [
    "compute the 10th fibonacci number",
    "what is the factorial of 6",
    "compute 2 to the power of 10",
]
SCRIPT_ROUND_2 = [
    "compute the 15th fibonacci number",    # same shape as fib
    "what is the factorial of 8",           # same shape as factorial
    "compute 3 to the power of 5",          # same shape as power
]


async def _ask(rt: Runtime, text: str) -> None:
    """Send one intent through the runtime and print the answer."""
    response = await rt.handle_intent(Intent(text=text, source="example"))
    # `last_reasoner` is persisted in latent state each turn.
    tier = rt.latent.get("last_reasoner")
    print(f"  >>> {text}")
    print(f"      [{tier}] {response.text}")


def _print_library(rt: Runtime) -> None:
    """Dump each promoted primitive with its invocation count."""
    prims = rt.library.list()
    if not prims:
        print("  (empty)")
        return
    for p in prims:
        print(f"  {p.name:36s} entry={p.entrypoint:10s} calls={p.invocations}")


async def main() -> None:
    # A fresh data dir guarantees the library starts empty for the demo.
    shutil.rmtree(DATA_DIR, ignore_errors=True)

    # `Runtime` wires up the full stack: bus, world model, memory, reasoners
    # (fast heuristic + synthesis + deep-sim), router, verifier, library,
    # consolidator, curiosity. No API keys needed — stubs fill in where a
    # real LLM would normally sit.
    rt = Runtime(root=DATA_DIR)
    await rt.start()

    print("=" * 60)
    print("MENTE coding agent — synthesis + library reuse")
    print("=" * 60)

    print("\n-- round 1: first time we see each shape (synthesis path) --")
    for line in SCRIPT_ROUND_1:
        await _ask(rt, line)

    print("\n-- library after round 1 --")
    _print_library(rt)

    print("\n-- round 2: same shapes, different inputs (library reuse) --")
    for line in SCRIPT_ROUND_2:
        await _ask(rt, line)

    # Each of the three primitives should show invocations == 2 because the
    # `SynthesisReasoner` recognized a matching source hash and incremented
    # the counter instead of adding a second entry.
    print("\n-- library after round 2 (invocations should have incremented) --")
    _print_library(rt)

    # Show the consolidator's rolling digest: how many responses, routing mix.
    print("\n-- consolidation digest --")
    digest = rt.consolidator.consolidate()
    for key, val in digest.items():
        print(f"  {key}: {val}")

    await rt.shutdown()
    print("\nDone. State persisted in:", DATA_DIR)


if __name__ == "__main__":
    asyncio.run(main())
