# Quickstart

A 15-minute tour of MENTE. You'll install it, chat with it, watch it synthesize
code, and confirm that its state survives a restart.

## Prerequisites

- Python 3.11+
- A checkout of the repository (no dependencies required — MENTE's Phase 1
  runs on the stdlib)

## Install

Two ways to get started, from least to most permanent:

```bash
# Option A — no install, use the launcher in the checkout
./mente --help

# Option B — install from PyPI (coming with the first release)
pip install mente
mente --help
```

The rest of this page uses `./mente` since it works immediately after cloning.

## First session: `./mente run`

Launch the interactive REPL:

```bash
./mente reset      # start clean so output matches below
./mente run
```

You'll see the banner and a prompt:

```
   _      ___   ___
  /_\    | _ \ |_ _|   /_\
 / _ \   |   /  | |   / _ \
/_/ \_\  |_|_\ |___| /_/ \_\

  persistent, event-driven reasoning process

booted. 3 reasoners, 4 tools.
type a message. /help for commands. /quit to exit.

you>
```

Try this scripted sequence. Expected output is shown inline — it's
deterministic because Phase 1 uses stdlib-only embeddings and templated
synthesis.

### 1. Identity and clock

```
you> hello
mente[fast.heuristic]> Hi — I'm online.

you> who am I?
mente[fast.heuristic]> You are Jasper.

you> what time is it?
mente[fast.heuristic]> It is 2026-04-21T15:42:07.
```

The user's name comes from a `Belief` that the CLI seeds on startup (it
reads `$USER`). The clock answer is a real tool call — see `/bus` to
verify.

### 2. Memory

```
you> remember that redis uses an append-only file for durability
mente[fast.heuristic]> Noted: redis uses an append-only file for durability.

you> remember that postgres uses write-ahead logging
mente[fast.heuristic]> Noted: postgres uses write-ahead logging.

you> what do you remember?
mente[fast.heuristic]> I remember: redis uses an append-only file for durability;
postgres uses write-ahead logging.
```

Notes go to SQLite-backed slow memory and to the semantic-memory vector
store in one hop.

### 3. Semantic search

```
you> what do you know about databases?
mente[fast.heuristic]> About 'databases': redis uses an append-only file for
durability (score 0.14); postgres uses write-ahead logging (score 0.11).
```

Neither note contains the word "databases". The hits come from
`HashEmbedder` — character n-grams that give fuzzy lexical matches
without a language model.

### 4. Program synthesis

```
you> compute the 10th fibonacci number
mente[specialist.synthesis]> fib(n=10) = 55

you> /library
  lib.fib.3ddeaa  entry=fib  calls=1
```

The synthesizer recognized the shape, emitted Python, ran it in a
subprocess sandbox with a disallowed-AST gate, verified the output, and
promoted the function to the permanent library. Ask again and it skips
synthesis — it's a registered tool now.

### 5. Self-query

```
you> your reasoners
mente[fast.heuristic]> I have 3 reasoners loaded: fast.heuristic,
specialist.synthesis, deep.sim.

you> how many turns have you handled?
mente[fast.heuristic]> I have handled 6 turns so far.
```

The `SelfModel` reads `LatentState` and the reasoner roster. The fast
reasoner has a pattern that dispatches introspective questions to it.

## `./mente demo`

If you'd rather see everything at once, run:

```bash
./mente reset
./mente demo
```

This plays an 11-step scripted sequence: greeting, identity, clock,
memory capture, semantic recall, two computation intents (fibonacci and
factorial, both synthesized), one escalation to the deep-tier stub, and
two self-queries. At the end it prints the synthesized-primitive library
so you can see that the functions stuck.

## Reset and state persistence

MENTE writes everything to `.mente/` under the project directory:

```
.mente/
  episodic.sqlite    append-only event log
  semantic.sqlite    vector store for semantic recall
  latent.json        checkpointed latent state (turns, last intent, digest)
  library.json       verified synthesized primitives
```

Latent state survives across invocations. To prove it, run two short
sessions back-to-back:

```bash
# Session 1
./mente reset
printf "remember that coffee helps my focus\n/quit\n" | ./mente run
# => Noted: coffee helps my focus.

# Session 2 — a brand-new process
printf "what do you remember?\n/quit\n" | ./mente run
# => I remember: coffee helps my focus.
```

The second process reloaded `latent.json`, reopened the SQLite stores,
and answered correctly. Nothing was in memory between the two runs.

To wipe state and start over:

```bash
./mente reset
```

This removes every `.mente*` directory in the project root.

## Next steps

- [Build an agent](build-an-agent.md) — a ~80-line journal agent that
  registers a custom tool, subscribes to intent events, and reuses
  MENTE's memory surface.
- [Extending MENTE](../extending.md) — protocol signatures and minimal
  implementations for adding your own reasoners, tools, specialists,
  embedders, synthesizers, and verifiers.
- [Architecture](../architecture.md) — why the system is shaped this way.
- [API reference](../reference/) — every public symbol.
