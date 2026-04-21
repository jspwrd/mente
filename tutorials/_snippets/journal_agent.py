"""A simple journal agent built on MENTE in ~80 lines.

Run it directly:

    python docs/tutorials/_snippets/journal_agent.py

The agent:
  - registers a `journal.add` tool that writes free-text diary entries into
    MENTE's semantic memory
  - asserts a Belief about who the user is so the world model is seeded
  - subscribes to `intent.*` events and auto-detects diary-shaped input,
    routing it to the tool without the user having to say "save this"
  - answers reflective queries via MENTE's built-in semantic search

It uses only the public surface: Runtime, Intent, Belief, plus the tool
registry decorator. No API keys required — the HashEmbedder handles
semantic recall and the default fast/deep reasoners handle dispatch.
"""
from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

from mente.runtime import Runtime
from mente.types import Belief, Event, Intent


# --- heuristics for spotting diary-shaped input ---------------------------
DIARY_TRIGGERS = ("today", "tonight", "this morning", "dear diary", "i felt", "i feel")


def looks_like_diary(text: str) -> bool:
    lower = text.lower()
    return any(t in lower for t in DIARY_TRIGGERS)


async def build_agent(root: Path) -> Runtime:
    rt = Runtime(root=root)
    await rt.start()

    # Seed a world-model belief so the agent knows who it's journaling for.
    await rt.world.assert_belief(
        Belief(entity="user", attribute="name", value="Journaler"),
    )
    await rt.world.assert_belief(
        Belief(entity="user", attribute="role", value="journal keeper"),
    )

    # Register a custom tool. MENTE's tool registry handles typing + cost
    # bookkeeping; the router will see it on the registry like any other.
    @rt.tools.register(
        "journal.add",
        "Append a diary entry to semantic memory.",
        returns="int",
        est_cost_ms=3.0,
    )
    async def _journal_add(entry: str) -> int:
        # Store twice: once under "journal" for domain-specific recall, once
        # under "note" so MENTE's built-in "what do you know about X?" path
        # (which queries kind="note") can also surface these entries.
        rt.semantic_mem.remember(entry, kind="journal")
        rid = rt.semantic_mem.remember(entry, kind="note")
        print(f"  [journal.add] stored entry #{rid}")
        return rid

    # Subscribe to intent events. When a user utterance looks like a diary
    # entry we quietly mirror it into the journal tool — no custom router
    # needed, the bus fans it out for us.
    async def auto_capture(event: Event) -> None:
        text = event.payload.get("text", "")
        if looks_like_diary(text):
            await rt.tools.invoke("journal.add", entry=text)

    rt.bus.subscribe("intent.*", auto_capture, name="journal.auto_capture")
    return rt


async def main() -> None:
    # Use a throwaway state dir so the demo is reproducible.
    root = Path(tempfile.mkdtemp(prefix="mente-journal-"))
    try:
        rt = await build_agent(root)

        print("=== journal agent online ===")
        print(f"user belief: {rt.world.entity('user')}")
        print(f"tools:        {[t.name for t in rt.tools.list()]}")

        print("\n--- writing diary entries ---")
        for line in [
            "Today I shipped the first working draft of the journal agent.",
            "Tonight I felt tired but grateful — the bus subscription trick worked.",
            "This morning I debugged a tricky semantic-search ranking issue.",
        ]:
            print(f"> {line}")
            await rt.handle_intent(Intent(text=line))

        print("\n--- reflective recall ---")
        # MENTE's fast-tier reasoner recognizes "what do you know about X"
        # and routes it to the semantic search tool automatically.
        question = "what do you know about debugging?"
        print(f"> {question}")
        response = await rt.handle_intent(Intent(text=question))
        print(f"mente> {response.text}")

        print("\n--- direct semantic-memory query ---")
        hits = rt.semantic_mem.search("shipping code", k=2, kind="journal")
        for h in hits:
            print(f"  score={h['score']:+.3f}  {h['text']}")

        await rt.shutdown()
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
