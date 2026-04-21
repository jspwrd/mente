"""Research agent example: semantic memory ingestion + search.

Demonstrates MENTE's semantic memory: a hash-embedder (character n-gram
feature hashing, stdlib-only) backs cosine-similarity search over a
corpus of notes. It is not a transformer embedding — 'car' and
'automobile' won't cluster — but shared morphology ('deploy' vs
'deployment') does, which is enough to show the retrieval path.

The runtime exposes two routes into the same store:
  - `rt.semantic_mem.remember(text, kind=...)` — direct ingestion.
  - `await rt.handle_intent(Intent(text="remember that ..."))` — via the
    fast reasoner, which writes through the `memory.note` tool.

Run:
    python examples/research_agent.py
"""
from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from mente.runtime import Runtime  # noqa: E402
from mente.types import Intent  # noqa: E402


DATA_DIR = _ROOT / ".mente-example-research"


# Ten fake research notes across a few clustered topics.
NOTES = [
    # distributed systems cluster
    "raft is a consensus protocol with a leader election phase",
    "paxos requires a majority quorum to commit a decision",
    "gossip protocols spread state by random pairwise exchange",
    # databases cluster
    "postgres uses multi-version concurrency control for snapshot isolation",
    "sqlite stores the entire database as a single file on disk",
    "redis persists with append-only log or snapshot files",
    # ML / LLM cluster
    "transformers scale quadratically with context length",
    "mixture-of-experts routes tokens to a sparse subset of parameters",
    # hardware cluster
    "raspberry pis are memory-bandwidth-bound for large model inference",
    "gpus deliver high throughput but their host bandwidth is often the bottleneck",
]


QUERIES = [
    "consensus protocol",       # should pull raft / paxos / gossip
    "database persistence",     # should pull redis / sqlite / postgres
    "language model context",   # should pull transformer / moe notes
    "memory bandwidth",         # should pull pi / gpu
    "leader election",          # narrow — should strongly prefer raft
]


async def main() -> None:
    # Fresh data dir — we want deterministic output from the embedder.
    shutil.rmtree(DATA_DIR, ignore_errors=True)

    rt = Runtime(root=DATA_DIR)
    await rt.start()

    print("=" * 60)
    print("MENTE research agent — semantic memory ingestion + search")
    print("=" * 60)

    # Half of the notes go through the intent path (so they also show up
    # in episodic memory and in the consolidation digest); the rest go
    # through the direct API for speed.
    print("\n-- ingesting notes --")
    for i, note in enumerate(NOTES):
        if i % 2 == 0:
            # Intent-shaped ingestion exercises the full pipeline.
            await rt.handle_intent(Intent(text=f"remember that {note}", source="example"))
        else:
            # Direct ingestion — semantically equivalent, less overhead.
            rt.semantic_mem.remember(note, kind="note")
        print(f"  [{i + 1:2d}/10] {note}")

    # Run a handful of queries. The hash-embedder gives high similarity
    # for notes that share 3/4-gram structure with the query.
    print("\n-- semantic queries --")
    for q in QUERIES:
        hits = rt.semantic_mem.search(q, k=3, kind="note")
        print(f"\n  query: {q!r}")
        if not hits:
            print("    (no results)")
            continue
        for h in hits:
            print(f"    score={h['score']:.3f}  {h['text']}")

    # The consolidator's digest summarizes what the session looked like —
    # accept rate is the verifier's pass-rate over responses; note_count
    # reflects facts that flowed through the episodic log.
    print("\n-- consolidation digest --")
    digest = rt.consolidator.consolidate()
    print(f"  note_count:       {digest['note_count']}")
    print(f"  total_responses:  {digest['total_responses']}")
    print(f"  accept_rate:      {digest['accept_rate']}")
    print(f"  avg_verdict:      {digest['avg_verdict_score']}")
    print(f"  by_reasoner:      {digest['by_reasoner']}")

    await rt.shutdown()
    print("\nDone. State persisted in:", DATA_DIR)


if __name__ == "__main__":
    asyncio.run(main())
