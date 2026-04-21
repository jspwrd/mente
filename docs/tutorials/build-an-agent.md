# Build an agent

You'll build a **journal agent** on top of MENTE: a small domain-specific
agent that captures free-text diary entries, stores them with semantic
search, and answers reflective queries. End-to-end: under 100 lines.

The full runnable source is at
[`_snippets/journal_agent.py`](_snippets/journal_agent.py). Run it:

```bash
python docs/tutorials/_snippets/journal_agent.py
```

## What you're reusing

Most of the wiring comes from MENTE. You don't have to write:

- a message bus — `EventBus` is already on the runtime
- intent routing — `Router` picks the right reasoner by predicted
  cost/confidence
- a tool registry — `@rt.tools.register` gives you a typed tool with
  cost estimates in four lines
- a verifier — `Verifier` returns a structured verdict on every response
- persistent memory — `SlowMemory` (SQLite episodes) and
  `SemanticMemory` (vector store) are ready to use
- a world model — `WorldModel` is an EAV blackboard that emits
  `state.*` events on every write

Your job is to plug domain-specific logic into those hooks.

## Scenario

The journal agent does three things:

1. **Captures diary-shaped input automatically**. When the user says "Today
   I shipped…" we don't want to require them to type
   `remember that …`. We'll subscribe to `intent.*` events and mirror
   diary-shaped utterances into our own tool.
2. **Stores entries with semantic search**. Uses MENTE's `semantic_mem`
   (HashEmbedder + cosine search) — no API keys needed.
3. **Answers reflective queries**. "What do you know about debugging?" is
   already a shape the fast-tier reasoner handles via `memory.search`. By
   writing journal entries under `kind="note"` as well, we inherit that
   behaviour for free.

## Walkthrough

The full file is under 100 lines. We'll go piece by piece.

### 1. Instantiate the runtime

```python
from mente.runtime import Runtime
from mente.types import Belief, Event, Intent

rt = Runtime(root=root)
await rt.start()
```

The runtime builds the bus, default reasoners (fast + synthesis + deep
stub), router, verifier, memory surfaces, and latent-state checkpointer.
It also registers the built-in tools (`clock.now`, `memory.note`,
`memory.recall`, `memory.search`).

### 2. Seed a world-model belief

```python
await rt.world.assert_belief(
    Belief(entity="user", attribute="name", value="Journaler"),
)
await rt.world.assert_belief(
    Belief(entity="user", attribute="role", value="journal keeper"),
)
```

Beliefs give reasoners and the verifier grounded context. Every write
emits a `state.<entity>.<attribute>` event — other subscribers can react
without polling.

### 3. Register a custom tool

```python
@rt.tools.register(
    "journal.add",
    "Append a diary entry to semantic memory.",
    returns="int",
    est_cost_ms=3.0,
)
async def _journal_add(entry: str) -> int:
    rt.semantic_mem.remember(entry, kind="journal")
    rid = rt.semantic_mem.remember(entry, kind="note")
    return rid
```

`rt.tools.register` reads the function signature, stamps on the
description, return type, and cost estimate, and hands the spec to the
metacog so the router can account for it. Reasoners can call
`rt.tools.invoke("journal.add", entry=...)` and your tool is just
another node.

> We write twice — once under `kind="journal"` for domain-specific recall
> and once under `kind="note"` so the built-in `"what do you know
> about X?"` fast-path (which filters on `kind="note"`) finds them too.

### 4. Subscribe to intent events

```python
DIARY_TRIGGERS = ("today", "tonight", "this morning",
                  "dear diary", "i felt", "i feel")

def looks_like_diary(text: str) -> bool:
    lower = text.lower()
    return any(t in lower for t in DIARY_TRIGGERS)

async def auto_capture(event: Event) -> None:
    text = event.payload.get("text", "")
    if looks_like_diary(text):
        await rt.tools.invoke("journal.add", entry=text)

rt.bus.subscribe("intent.*", auto_capture, name="journal.auto_capture")
```

The bus fans out every intent to every pattern-matched subscriber. We
don't replace the router; we just tap the event stream. This is the
idiom for adding reflex-level behaviour without rewriting the cortex.

### 5. Drive it

```python
response = await rt.handle_intent(
    Intent(text="Today I shipped the first working draft.")
)
# auto_capture fires → journal.add is invoked → entry stored.

response = await rt.handle_intent(Intent(text="what do you know about shipping?"))
# fast.heuristic matches the "what do you know about X?" pattern →
# invokes memory.search → returns the stored entry.
```

That's the whole loop.

## Full source

```python
--8<-- "docs/tutorials/_snippets/journal_agent.py"
```

If your doc toolchain doesn't support the `--8<--` include syntax, open
[`_snippets/journal_agent.py`](_snippets/journal_agent.py) directly.

## What MENTE's bus / router / verifier save you

Rolling this from scratch would require, at minimum:

| Concern                | What MENTE provides                             |
| ---------------------- | ---------------------------------------------- |
| Pub/sub with wildcards | `EventBus.subscribe("intent.*", handler)`     |
| Tiered routing         | `Router` with cost/confidence trade-off        |
| Confidence escalation  | Automatic fallback fast → deep on low-conf    |
| Structured verdicts    | `Verifier` on every response                   |
| Episodic audit log     | `SlowMemory` (SQLite, `state.*` + `response.*`)|
| Vector recall          | `SemanticMemory` (cosine search, persistent)   |
| Persistent state       | `LatentState.checkpoint()` every turn          |
| Tool-call bookkeeping  | `ToolRegistry` with typed params + cost        |
| Self-introspection     | `SelfModel.answer("what are your tools?")`     |

For a domain-specific agent you wire up the *distinctive* part
(`journal.add`, the diary trigger heuristic, the world-model beliefs
for this user) and inherit the rest.

## Next steps

- [Extending MENTE](../extending.md) — if you want to plug in a real
  reasoner, embedder, or verifier, these are the protocol shapes.
- [Architecture](../architecture.md) — how the pieces fit together.
- [API reference](../reference/index.md) — full signatures for every surface.
