"""Personal-organizer example: world model + remember/recall + curiosity.

Demonstrates three interacting surfaces:
  1. World model — beliefs keyed by (entity, attribute) that any reasoner
     can read. We seed it with the user's identity and a couple of
     domain entities (calendar, projects).
  2. Episodic + semantic memory — via "remember that ..." intents routed
     through the fast reasoner to the `memory.note` tool.
  3. Curiosity loop — on idle, it scans the world model and latent state
     for gaps (e.g. entities with only one attribute) and emits
     self-generated intents. We force a tick with `idle_threshold_s=0`
     so we don't have to wait.

Run:
    python examples/personal_org.py
"""
from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from mente.runtime import Runtime  # noqa: E402
from mente.types import Belief, Intent  # noqa: E402

DATA_DIR = _ROOT / ".mente-example-personal"


async def main() -> None:
    shutil.rmtree(DATA_DIR, ignore_errors=True)

    rt = Runtime(root=DATA_DIR)
    await rt.start()

    print("=" * 60)
    print("MENTE personal organizer — world model + memory + curiosity")
    print("=" * 60)

    # -- 1. seed the world model -------------------------------------------
    # Beliefs are (entity, attribute, value) triples. Each assert emits a
    # `state.<entity>.<attribute>` event on the bus, so any subscriber can
    # react to changes.
    print("\n-- seeding world model --")
    for belief in [
        Belief(entity="user", attribute="name", value="Alex"),
        Belief(entity="user", attribute="timezone", value="America/Los_Angeles"),
        Belief(entity="calendar", attribute="next_event", value="standup 10am"),
        Belief(entity="projects", attribute="active", value="mente-launch"),
    ]:
        await rt.world.assert_belief(belief)
        print(f"  {belief.entity}.{belief.attribute} = {belief.value}")

    # -- 2. write a few memories via the intent path -----------------------
    # The fast reasoner matches "remember that ..." and calls `memory.note`,
    # which writes to both episodic (SQLite) and semantic (hash-embedded) stores.
    print("\n-- capturing memories --")
    facts = [
        "remember that the quarterly review is on Friday",
        "remember that I promised Sam a draft of the README",
        "remember that the oss launch blocks on the example gallery",
    ]
    for f in facts:
        r = await rt.handle_intent(Intent(text=f, source="example"))
        print(f"  {f}  ->  {r.text}")

    # Recall path round-trips through `memory.recall`.
    r = await rt.handle_intent(Intent(text="what do you remember?", source="example"))
    print(f"\n-- recall --\n  {r.text}")

    # -- 3. force the curiosity loop --------------------------------------
    # Setting idle_threshold_s=0 removes the wait; `tick()` then scans for
    # gaps (e.g. entities with a single attribute) and publishes
    # `curiosity.generate` events. Runtime subscribes these and routes
    # them back through `handle_intent` with source='curiosity'.
    print("\n-- forcing a curiosity tick (idle_threshold=0) --")
    rt.curiosity.idle_threshold_s = 0
    emitted = await rt.curiosity.tick()
    # Give the bus dispatch one turn to deliver the self-generated intents.
    await asyncio.sleep(0.05)
    if not emitted:
        print("  (no gaps detected — add fewer attributes to an entity to trigger)")
    else:
        print("  self-generated intents:")
        for text in emitted:
            print(f"    * {text}")

    # -- 4. self-model description ----------------------------------------
    # The self-model is a queryable surface over the runtime's own state:
    # reasoners loaded, tools registered, turn count, last digest. The
    # fast reasoner forwards meta-questions to `SelfModel.answer`, which
    # routes on keywords (reasoner / tool / turn / doing).
    print("\n-- self-model: 'your reasoners?' --")
    r = await rt.handle_intent(Intent(text="your reasoners?", source="example"))
    print(f"  {r.text}")

    print("\n-- self-model: 'your tools?' --")
    r = await rt.handle_intent(Intent(text="your tools?", source="example"))
    print(f"  {r.text}")

    print("\n-- self-model: 'what have you been doing?' --")
    # A digest must exist before this answers meaningfully.
    rt.consolidator.consolidate()
    r = await rt.handle_intent(Intent(text="what have you been doing?", source="example"))
    print(f"  {r.text}")

    await rt.shutdown()
    print("\nDone. State persisted in:", DATA_DIR)


if __name__ == "__main__":
    asyncio.run(main())
