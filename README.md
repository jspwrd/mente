# ARIA

A persistent, event-driven reasoning process. Seed of a cognitive architecture
with heterogeneous reasoner tiers, semantic/episodic memory, program synthesis
with a verified-primitive library, a self-model, a curiosity loop, and a
distributed bus for specialist peers.

Runs on stdlib Python 3.11+ — no install, no dependencies required.

## Quickstart

```bash
./aria           # interactive REPL — try "compute the 15th fibonacci number"
```

That's it. State persists under `.aria/` in the project directory.

## All subcommands

```bash
./aria run        # interactive REPL (default)
./aria demo       # scripted walkthrough
./aria federated  # hub + specialist peer co-hosted, real TCP bus between them
./aria test       # smoke tests (bus, synthesis, semantic memory)
./aria reset      # wipe all .aria* state directories
./aria --help     # full help
```

Inside the REPL, slash commands let you peek at internals:

```
/state     current latent state
/library   synthesized primitives (persistent, reused)
/bus       last 20 events on the bus
/digest    force a consolidation digest now
/help      list commands
/quit      exit
```

## What it can do

Talk to it. Some things the built-in fast-tier heuristics recognize:

- **identity**: `hello`, `who am I?`, `what time is it?`
- **memory**: `remember that X`, `what do you remember?`, `what do you know about X?`
- **computation**: `compute the Nth fibonacci number`, `what is the factorial of N`, `N to the power of M` — synthesizes code, runs it sandboxed, promotes it to a permanent tool
- **introspection**: `what are you?`, `how many turns have you handled?`, `what have you been doing?`

Anything it can't pattern-match falls through to the deep tier (a stub by
default — see below for wiring real Claude).

## Run with real Claude

```bash
pip install 'anthropic>=0.40.0'
export ANTHROPIC_API_KEY=sk-ant-...
./aria
```

The deep tier auto-detects the key and swaps in `claude-opus-4-7` with
adaptive thinking + prompt caching. Every other tier stays exactly the same.

## Multi-process peers

`./aria federated` runs a hub + a math specialist peer in the same process
but with a real TCP bus between them — the cleanest way to see federation
working in one command.

If you want them in separate terminals:

```bash
# Terminal 1 (peer)
./aria peer --port 7722

# Terminal 2 (hub)
ARIA_BUS_ROLE=hub ARIA_BUS_PORT=7722 ./aria run
```

## Layout

```
src/aria/
  cli.py          entry point — all commands
  runtime.py      the event-loop process that owns the whole stack
  bus.py          async pub/sub with wildcard topics
  transport.py    pluggable (in-proc / TCP); add NATS etc. as a third option
  world_model.py  entity-attribute-value blackboard
  memory.py       fast (TTL) + slow (SQLite episodic) memory tiers
  embeddings.py   semantic memory with cosine search (n-gram hash embedder)
  state.py        persistent latent state, checkpointed every turn
  tools.py        typed tool registry
  reasoners.py    fast heuristic + deep stub; LLM reasoner lives in llm.py
  llm.py          AnthropicReasoner (Claude Opus 4.7, adaptive thinking, caching)
  metacog.py      predicts per-reasoner confidence/cost per intent
  router.py       dispatches by predicted cost/confidence, escalates on low-conf
  verifier.py     step-wise verdict with structured reasons
  specialists.py  MathSpecialist (safe arithmetic evaluator)
  synthesis.py    program synthesis + verified-primitive library
  discovery.py    peer capability announcements + federated routing
  curiosity.py    idle-time self-prompting from world-model gaps
  consolidator.py "sleep cycle" that distills episodes into digests
  self_model.py   structured, queryable self-representation
```

## Troubleshooting

- **`python 3.11+ required`** — your system Python is older. Install 3.11+ or
  set `ARIA_PYTHON=/path/to/python3.11`.
- **Hangs on Ctrl-C in federated mode** — the REPL's `input()` blocks the
  event loop cleanup; press Enter first, then `/quit`.
- **`port already in use`** — another ARIA hub is up; change `--port` or
  `./aria reset` and retry.
