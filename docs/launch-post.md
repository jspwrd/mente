# I got tired of LangChain, so I wrote a 3,000-line cognitive-architecture framework

*Draft for HN / lobste.rs / r/LocalLLaMA. Version: first cut, not yet posted.*

---

Most agent frameworks feel like Java to me. LangChain, LangGraph, CrewAI —
they wrap one LLM in ten layers of abstraction and ask you to learn a new
vocabulary (tools, chains, runnables, agents, executors, callbacks,
retrievers, memory types, output parsers) before you can do anything. The
stack traces are uninterpretable. The examples always work. Your code
never does.

I wanted something I could read in one sitting.

[**mente**](https://github.com/jspwrd/mente) (Latin / Italian / Spanish for
"mind") is a ~3,000-line Python framework for building persistent, event-
driven agents. Stdlib-only core. Optional extras for real LLMs and
embeddings. No runtime dependencies in the base install. Every module has a
docstring explaining what's shipping today (Phase 1) vs what's planned
(Phase 2), so you always know what's load-bearing and what's a stub.

## What's actually different

Frameworks in this space are mostly "single LLM + orchestration." mente
is a **population of reasoners with a bus between them**:

```
     ┌──────────── Event Bus ────────────┐
     │          (in-proc / TCP)          │
     └──┬─────────┬──────────┬──────────┬┘
        │         │          │          │
   ┌────▼──┐  ┌───▼───┐  ┌───▼────┐ ┌───▼───┐
   │ Fast  │  │ Math  │  │ Code   │ │ Deep  │
   │heuris-│  │special│  │special │ │ LLM   │
   │ tic   │  │ ist   │  │ ist    │ │(Claude│
   └───────┘  └───────┘  └────────┘ └───────┘
```

A **metacog** module predicts — before routing — which reasoner is most
confident on this intent and at what cost. A **router** picks the cheapest
reasoner above a confidence threshold and escalates if the response comes
back weak. A **step-wise verifier** scores every response and can reject.
This isn't novel research — it's the kind of structure you'd build yourself
after six months of fighting a monolithic-LLM agent, cleaned up and named.

The bits that make it fun:

- **Program synthesis grows the tool library at runtime.** Ask it to
  compute the 15th Fibonacci number. It synthesizes a function, validates
  the AST, runs it in a subprocess sandbox, and — if it works — promotes it
  to a permanent, typed tool. Next time you ask, the primitive is just
  there. The library persists to disk.
- **Tiered memory.** Latent state (JSON, checkpointed every turn), fast
  TTL scratchpad, slow SQLite episodic log, and a semantic store with
  cosine search. The default embedder is an offline character n-gram
  hasher (no dependencies). Drop in `mente[embeddings-local]` for
  sentence-transformers or `mente[embeddings]` for Voyage.
- **Curiosity loop.** Background task that inspects the world model for
  gaps during idle time and self-prompts. Catches things you didn't ask
  about.
- **Federation.** Specialist peers can live in separate processes on a
  shared TCP bus. `./mente federated` runs a hub + math specialist
  co-hosted with a real TCP connection between them. One command, real
  distributed cognition.
- **Self-model.** The system has a queryable structured representation of
  itself: which reasoners are loaded, what tools exist, what it's been
  doing. Ask it "what have you been doing?" and it answers from actual
  state, not confabulation.

## What it looks like

```bash
pip install mente                      # or: uv add mente
./mente                                # drops you into a REPL
```

```
you> hello
mente[fast.heuristic]> Hi — I'm online.
you> remember that redis uses AOF for persistence
mente[fast.heuristic]> Noted: redis uses AOF for persistence.
you> what do you know about databases?
mente[fast.heuristic]> About 'databases': redis uses AOF for persistence (score 0.18); ...
you> compute the 15th fibonacci number
mente[specialist.synthesis]> fib(n=15) = 610
you> /library
  lib.fib.3ddeaa  entry=fib  calls=1
you> /quit
```

Every turn, the REPL tells you which reasoner handled it. The `/library`
command shows primitives you've synthesized across sessions. State
persists. Run it again and the turn counter picks up where it left off.

## What's explicitly a stub

I'd rather be honest than oversell:

| Component | Shipping | Plug-in |
|---|---|---|
| Embedder | char-n-gram hash (offline) | Voyage API or local sentence-transformers |
| Deep reasoner | simulated latency (offline) | Claude Opus 4.7 with adaptive thinking + prompt caching |
| Synthesizer | templated (fib / factorial / pow) | LLM-authored (Claude writes the function) |
| Verifier | hand-coded heuristics | process reward model (Phase 2) |
| Metacog | hand-coded pattern coverage | trained head (Phase 2) |

Each is behind a Protocol so you can swap it without touching the rest.

## Numbers

- 3,000 lines of Python, 22 modules, 0 runtime deps in the core install.
- 392 tests, CI on Python 3.11/3.12/3.13 × ubuntu/macos.
- First `pip install` → `./mente run` works in under 5 seconds.

## Who this is for

- People who want to build agents without buying into a framework's
  worldview.
- People who found LangChain frustrating specifically because of the
  abstractions, not the LLM.
- Researchers who want a minimal runtime to experiment with routing,
  synthesis, or federation ideas on.

It is **not** for people who want a polished production stack — 0.1.0 is
alpha, the API will move.

## Design notes + criticism welcome

The architecture essay lives at
[`docs/architecture.md`](https://jspwrd.github.io/mente/architecture/) if
you want the subsystem-by-subsystem walkthrough.

I'm especially interested in feedback on:

- Whether "metacog + router" is a useful abstraction or a premature
  abstraction of what should just be code in each caller.
- Whether the program-synthesis-promotes-to-library pattern is actually a
  good idea in practice, or collects garbage faster than value.
- The naming of primitives (reasoner, specialist, verifier, consolidator).
  Some of these overlap with terms in other frameworks and I'd rather
  align than diverge unless the difference is real.

GitHub: https://github.com/jspwrd/mente
Docs: https://jspwrd.github.io/mente/
