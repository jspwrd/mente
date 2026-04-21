# ARIA

A persistent, event-driven reasoning process. Seed of a cognitive architecture
with heterogeneous reasoner tiers, semantic/episodic memory, program synthesis
with a verified-primitive library, a self-model, a curiosity loop, and a
distributed bus for specialist peers.

Runs on stdlib Python 3.11+ — no install, no dependencies required.

## Features

- **Event bus at the center.** Every subsystem is a publisher/subscriber; no
  direct calls, no god-object runtime.
- **Heterogeneous reasoners.** Fast heuristics, specialists, and an optional
  deep Claude tier. The router picks per-intent based on predicted cost and
  confidence.
- **Memory that actually tiers.** TTL working memory, SQLite episodic store,
  and n-gram-hash semantic embeddings with cosine search.
- **Program synthesis.** Recognized patterns compile to sandboxed Python,
  verify, and graduate into a persistent primitive library.
- **Curiosity & consolidation.** Idle-time self-prompting from world-model
  gaps; a sleep-cycle that distills episodes into digests.
- **Federation.** Peer discovery and capability announcements over a real
  TCP bus — specialists can live in other processes or machines.

## Installation

```bash
pip install aria
```

Or run straight from a checkout — ARIA's core has zero runtime dependencies:

```bash
git clone https://github.com/example/aria
cd aria
./aria
```

## 10-line example

```python
import asyncio
from pathlib import Path

from aria.runtime import Runtime
from aria.types import Intent

async def main():
    rt = Runtime(root=Path(".aria"))
    response = await rt.handle_intent(Intent(text="compute the 15th fibonacci number"))
    print(response.text)

asyncio.run(main())
```

Or just drop into the REPL:

```
$ ./aria
aria> compute the 15th fibonacci number
610
aria> remember that my favorite number is 7
ok
aria> what do you remember?
- favorite number is 7
```

## How is it different?

| Capability                 | ARIA                              | LangChain                  | OpenClaw                  |
| -------------------------- | --------------------------------- | -------------------------- | ------------------------- |
| Core dependencies          | stdlib only                       | dozens                     | a few                     |
| Architecture style         | event bus, pub/sub                | chain/graph DSL            | scripted orchestrator     |
| Reasoner tiers             | fast + specialist + deep (tiered) | single LLM per call        | single LLM per call       |
| Memory                     | TTL + episodic + semantic         | vector store plugin        | context window            |
| Program synthesis          | first-class, verified library     | not a core concern         | not a core concern        |
| Federation / peers         | built-in (TCP bus, discovery)     | external infra             | external infra            |
| Runs offline / no API key  | yes (heuristic + synthesis tiers) | rarely                     | no                        |

## Where to next

- [Architecture](architecture.md) — the deep-dive with a Mermaid diagram.
- [Tutorials](tutorials/quickstart.md) — step-by-step from zero to a
  customized agent.
- [Reference](reference/index.md) — every module, auto-generated from
  docstrings.
