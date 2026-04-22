# FAQ

Short answers to the questions that come up most often.

## Is this production-ready?

No. MENTE is a 0.x project and the public API will move between minor
versions. Pin `mente==0.1.x` if you need stability; expect to read the
changelog when you upgrade. The test suite and CI matrix are serious, but
the shape of the framework itself is still settling.

## Why not just use LangChain or LangGraph?

They solve a different problem — orchestrating calls to one LLM through
layers of chains and graph state. MENTE is a *persistent reasoning process*:
a population of tiered reasoners cooperating through a bus, with tiered
memory, synthesized primitives, and a curiosity loop running in the
background. The README table under [Why mente](index.md) lays out the axis
differences.

The pragmatic version: when something goes wrong, count the lines of code
you have to read to debug it. LangChain's stack is ~200K LOC across many
packages. MENTE's core is ~3,000 LOC, stdlib-only, one directory.

## How do I run it without installing?

Clone and run the launcher:

```bash
git clone https://github.com/jspwrd/mente
cd mente
./mente
```

The `./mente` launcher resolves a Python 3.11+ interpreter (respecting
`MENTE_PYTHON` if set) and runs the CLI directly from the checkout. No
install, no venv, no dependencies — the core is stdlib-only.

## Does it work without Claude?

Yes. With no `ANTHROPIC_API_KEY` set, MENTE runs fully offline:

- `DeepSimulatedReasoner` stands in for the deep tier (stub latency, canned
  shape).
- `HashEmbedder` provides semantic search via character-n-gram hashing —
  fuzzy lexical matches, no model download.
- `TemplateSynthesizer` handles known synthesis patterns (fib, factorial,
  power) without an LLM.

Everything above the reasoner layer — router, memory, verifier, bus,
curiosity, consolidation — is identical in both modes.

## How do I integrate async and sync code?

The `Runtime` is async-native; reasoners, tools, and subscribers are all
coroutines. If you need to call sync code from a reasoner (a blocking I/O
library, a CPU-heavy helper), wrap it with `asyncio.to_thread`:

```python
import asyncio

async def my_reasoner(intent):
    result = await asyncio.to_thread(sync_function, intent.text)
    return result
```

The opposite direction — calling async MENTE code from a sync caller — is
what `asyncio.run(main())` handles in the README's ten-line example. Don't
call `asyncio.run` from inside a running loop; reuse the loop you're in.

## What's the license?

MIT. See [LICENSE](https://github.com/jspwrd/mente/blob/main/LICENSE) for
the full text. Use it, fork it, ship it — attribution appreciated, not
required beyond the license header.
