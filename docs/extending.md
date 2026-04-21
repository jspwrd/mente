# Extending MENTE

Every moving part of MENTE is a swap point. This guide shows the current
shape of each hook, a minimal implementation you can drop in today, and
how to wire it into a `Runtime`.

> **Phase 2 note.** A few of these surfaces — **Embedder**, **Synthesizer**,
> **Verifier** — are defined only as Phase 1 conventions today (a
> `Protocol` in one case, a dataclass with a known method name in the
> others). They are slated to be formalized into dedicated packages
> (`mente.embedders`, `mente.synthesizers`, `mente.verifiers`) in Phase 2 with
> the same shape you see below. The examples here use the current public
> API exactly as it lives on `main`.

## Add a reasoner

A reasoner is the primary swap point between MENTE and a real model —
Claude, local Llama, a specialist fine-tune, anything that can map an
intent to a response.

### Protocol

From [`mente/reasoners.py`](reference/reasoners.md):

```python
class Reasoner(Protocol):
    name: str
    tier: ReasonerTier  # "fast" | "deep" | "specialist"
    est_cost_ms: float

    async def answer(
        self, intent: Intent, world: WorldModel, tools: ToolRegistry
    ) -> Response: ...
```

### Minimal implementation

```python
from dataclasses import dataclass
from mente.types import Intent, ReasonerTier, Response
from mente.tools import ToolRegistry
from mente.world_model import WorldModel

@dataclass
class GreetingReasoner:
    name: str = "fast.greet"
    tier: ReasonerTier = "fast"
    est_cost_ms: float = 0.5

    async def answer(
        self, intent: Intent, world: WorldModel, tools: ToolRegistry
    ) -> Response:
        if intent.text.lower().startswith(("hi", "hello", "hey")):
            return Response(
                text="Greetings, traveler.",
                reasoner=self.name, tier=self.tier,
                confidence=0.99, cost_ms=self.est_cost_ms,
            )
        return Response(
            text="", reasoner=self.name, tier=self.tier,
            confidence=0.0, cost_ms=self.est_cost_ms,
        )
```

### Wiring

```python
rt = Runtime(root=Path(".mente"))
await rt.start()
rt.reasoners.append(GreetingReasoner())
# The runtime's router and metacog both reference rt.reasoners by
# identity — appending is enough. If you want your reasoner tried first,
# insert at index 0.
```

## Add a tool

Tools are plain async Python callables with a typed signature and a cost
estimate. They're invoked by reasoners or, as in the journal tutorial,
by bus subscribers.

### Shape

From [`mente/tools.py`](reference/tools.md):

```python
@dataclass
class ToolSpec:
    name: str
    description: str
    params: dict[str, str]  # filled from the function signature
    returns: str
    est_cost_ms: float
    fn: Callable[..., Awaitable[Any]]
```

Registration is a decorator on `ToolRegistry.register`.

### Minimal implementation

```python
@rt.tools.register(
    "weather.guess",
    "Deterministic dummy-weather helper for demos.",
    returns="str",
    est_cost_ms=1.0,
)
async def _guess(city: str) -> str:
    return f"{city} is probably overcast."
```

### Wiring

Registration *is* wiring. Once registered, reasoners can call it:

```python
result = await rt.tools.invoke("weather.guess", city="Utrecht")
# ToolResult(tool='weather.guess', ok=True, value='Utrecht is probably overcast.', ...)
```

## Add a specialist

A specialist is a reasoner with narrow competence and higher confidence
in its domain than the general deep tier. The router prefers specialists
when the metacog recognizes a matching domain pattern.

### Minimal implementation

```python
from dataclasses import dataclass
from mente.types import Intent, ReasonerTier, Response

@dataclass
class UpperCaseSpecialist:
    name: str = "specialist.upper"
    tier: ReasonerTier = "specialist"
    est_cost_ms: float = 1.0

    async def answer(self, intent, world, tools) -> Response:
        if "shout" not in intent.text.lower():
            return Response(
                text="", reasoner=self.name, tier=self.tier,
                confidence=0.0, cost_ms=self.est_cost_ms,
            )
        return Response(
            text=intent.text.upper(),
            reasoner=self.name, tier=self.tier,
            confidence=0.99, cost_ms=self.est_cost_ms,
        )
```

### Wiring

```python
rt.reasoners.append(UpperCaseSpecialist())
```

For the router's metacog to *prefer* your specialist over the deep tier
on matching intents, teach it about your domain. Today that means adding
a pattern to `mente.metacog._SPECIALIST_PATS` (keyed by a substring of
your reasoner's `name`). Phase 2 will formalize this as a
`SpecialistRegistry` entry alongside each reasoner.

## Add an embedder

### Protocol

From [`mente/embeddings.py`](reference/embeddings.md):

```python
class Embedder(Protocol):
    dim: int
    def embed(self, text: str) -> list[float]: ...
```

Phase 1 ships `HashEmbedder` (stdlib-only, character n-grams, no API key).
Phase 2 will move `Embedder` into a dedicated `mente.embedders` package
alongside adapters for sentence-transformers, Voyage, and OpenAI.

### Minimal implementation

```python
from dataclasses import dataclass
import math

@dataclass
class ConstantEmbedder:
    """Degenerate embedder — useful as a placeholder in tests."""
    dim: int = 8

    def embed(self, text: str) -> list[float]:
        n = 1.0 / math.sqrt(self.dim)
        return [n] * self.dim
```

### Wiring

```python
from mente.embeddings import SemanticMemory
rt.semantic_mem = SemanticMemory(
    db_path=rt.root / "semantic.sqlite",
    embedder=ConstantEmbedder(),
)
```

For a fresh runtime, pass the swapped embedder before calling
`rt.start()`, or construct the `Runtime` with an explicit
`semantic_mem=`. (Phase 2 adds a constructor hook so you don't have to
reassign.)

## Add a synthesizer

The synthesizer turns computation-shaped intents into Python source that
the synthesis reasoner then sandboxes, validates, and promotes into the
verified-primitive library.

### Shape

From [`mente/synthesis.py`](reference/synthesis.md). Phase 1 doesn't yet
have a formal `Protocol` — `TemplateSynthesizer` establishes the shape:

```python
class Synthesizer:
    def synthesize(self, intent_text: str) -> tuple[str, str, dict] | None:
        """Return (source, entrypoint, args) or None if we can't synthesize."""
```

Phase 2 will lift this into `mente.synthesizers` with a proper `Protocol`.

### Minimal implementation

```python
import re

_COUNT_RE = re.compile(r"count (?:the )?words in\s+(.+)$", re.I)

class CountWordsSynthesizer:
    def synthesize(self, intent_text: str):
        m = _COUNT_RE.search(intent_text)
        if not m:
            return None
        src = (
            "def count_words(s):\n"
            "    return len(s.split())\n"
        )
        return src, "count_words", {"s": m.group(1).strip()}
```

### Wiring

```python
from mente.synthesis import SynthesisReasoner

# Replace the default SynthesisReasoner's synthesizer after construction:
for r in rt.reasoners:
    if isinstance(r, SynthesisReasoner):
        r.synthesizer = CountWordsSynthesizer()
```

Or build your own `SynthesisReasoner` that delegates to a chain of
synthesizers, and append it to `rt.reasoners` alongside the default one.

## Add a verifier

The verifier is consulted after a reasoner produces a response. It
returns an accept/reject verdict plus structured reasons. Rejections
can trigger rework or escalation.

### Shape

From [`mente/verifier.py`](reference/verifier.md):

```python
@dataclass
class Verdict:
    accept: bool
    score: float
    reasons: list[str]

class Verifier:  # duck-typed today; becomes a Protocol in Phase 2
    def verify(
        self, intent: Intent, response: Response, world: WorldModel
    ) -> Verdict: ...
```

Phase 1 ships a heuristic `Verifier` dataclass; a formal `Protocol` will
live in `mente.verifiers` in Phase 2 alongside adapters for PRM-style
trained verifiers and formal checkers (SMT, type systems, test runners).

### Minimal implementation

```python
from dataclasses import dataclass
from mente.verifier import Verdict

@dataclass
class LengthGate:
    """Reject one-liners — demands at least N characters."""
    min_chars: int = 20

    def verify(self, intent, response, world) -> Verdict:
        if len(response.text) < self.min_chars:
            return Verdict(
                accept=False, score=0.1,
                reasons=[f"response shorter than {self.min_chars} chars"],
            )
        return Verdict(accept=True, score=0.9, reasons=["ok"])
```

### Wiring

```python
rt.verifier = LengthGate(min_chars=40)
```

Reassign before or after `rt.start()` — the runtime reads `self.verifier`
fresh on every turn.

## Where to go next

- [Architecture](architecture.md) — why the surfaces are shaped this way
- [API reference](reference/) — every public symbol with its signature
- [Build an agent](tutorials/build-an-agent.md) — a complete example
  that registers a tool and subscribes to intents
