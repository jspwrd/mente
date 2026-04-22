# Extending MENTE

Every moving part of MENTE is a swap point. This guide shows the current
shape of each hook, a minimal implementation you can drop in today, and
how to wire it into a `Runtime`.

> **Package layout.** The extension surfaces live in dedicated packages
> today: `mente.embedders`, `mente.synthesizers`, and `mente.verifiers`
> each expose a `Protocol` plus one or more shipped implementations.
> `Reasoner` lives in `mente.reasoners`. Specialists live in
> `mente.specialists`. The examples here use the current public API
> exactly as it lives on `main`. Planned improvements are noted inline
> and tracked in [ROADMAP.md](https://github.com/jspwrd/mente/blob/main/ROADMAP.md).

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

### How existing defaults constrain your plug-in

The stock roster is `[FastHeuristicReasoner, SynthesisReasoner,
CodeSpecialist, deep]`, where `deep` is `AnthropicReasoner` when an API
key is present and `DeepSimulatedReasoner` otherwise. The `Router`
escalates by `tier_order = {"fast": 0, "deep": 1, "specialist": 1}` and
picks the deeper tier when `response.confidence < min_confidence`
(default 0.7). Your reasoner MUST set `confidence=0.0` on "don't know"
so escalation fires — raising instead breaks the pipeline. It MUST also
use one of the three literal tier strings or the router's `tier_order`
lookup will `KeyError`. You're free to pick any `name` (used only as a
log/trace key), any `est_cost_ms` (hint, not a bound), and any
confidence policy above 0.0 — the `Metacog.estimate` defaults to 0.55
for unknown deep-tier reasoners and 0.1 for unknown fast/specialist
names, so a novel reasoner will still be considered, just not preferred.

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

### How existing defaults constrain your plug-in

`Runtime._register_default_tools` registers `clock.now`, `memory.note`,
`memory.recall`, and `memory.search`; `FastHeuristicReasoner` hard-codes
those names. Pick a name that won't collide — the registry is a flat
`dict[str, ToolSpec]` and a duplicate `register()` silently replaces the
earlier tool. Your `fn` MUST be async (`ToolRegistry.invoke` does
`await spec.fn(**kwargs)`) and MUST accept only keyword arguments that
map to the inspected signature; `invoke` wraps exceptions into
`ToolResult(ok=False, error=repr(e))` so raising is safe but consumed
silently by callers that only check `.value`. You're free to return any
JSON-serializable or in-memory Python object as `.value` — the registry
does not introspect it — and `est_cost_ms` is purely advisory.

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
a pattern to `mente.metacog._SPECIALIST_PATS`, keyed by a substring of
your reasoner's `name` — e.g. the `"math"` key matches any reasoner
whose name contains `"math"`. A public `SpecialistRegistry` surface is
not on the current roadmap; track
[ROADMAP.md § Medium-term](https://github.com/jspwrd/mente/blob/main/ROADMAP.md#medium-term-03x--05x)
for the trained metacog head that will eventually subsume it.

### How existing defaults constrain your plug-in

`MathSpecialist` and `CodeSpecialist` both return `confidence=0.0` when
their domain doesn't match — that's how the router falls through to the
deep tier. Your specialist MUST do the same, because `Metacog` only
marks `specialist` reasoners as preferred when `r.name.lower()`
substring-matches a `_SPECIALIST_PATS` key; if `name` has no matching
substring, the metacog predicts `confidence=0.1` and the router picks
the deep tier instead. You're free to do arbitrary work inside
`answer` (subprocess calls, tool invocations, network I/O) as long as
you obey the `Reasoner` protocol's async-safety and never-raise-on-
unknown-input rules.

## Add an embedder

### Protocol

From [`mente/embeddings.py`](reference/embeddings.md) (the canonical
definition lives in `mente.embedders.hashing` and is re-exported):

```python
class Embedder(Protocol):
    dim: int
    def embed(self, text: str) -> list[float]: ...
```

`HashEmbedder` (stdlib-only, character n-grams, no API key) is the
default. `mente.embedders` also ships `VoyageEmbedder` (behind the
`embeddings` extra) and `LocalEmbedder` (sentence-transformers, behind
the `embeddings-local` extra); both are lazily imported.

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
`semantic_mem=`. A constructor hook so you don't have to reassign is not
currently scheduled — open an issue if you need one.

### How existing defaults constrain your plug-in

`HashEmbedder` returns unit-norm vectors of length `dim` (or the zero
vector for empty input). `SemanticMemory._cosine` treats dot product as
cosine similarity and assumes both operands are unit-normalized. Your
embedder SHOULD return unit-norm vectors so similarity scores stay in
`[-1, 1]` and search rankings remain meaningful; returning a zero
vector for empty/invalid input is the documented escape hatch. Every
returned vector MUST have length `self.dim`, since vectors are stored
base64-encoded with no per-row length header and a dimension mismatch
surfaces only at `zip(..., strict=True)` time. `embed` MUST be
synchronous and cheap — `SemanticMemory.search` calls it inline on the
query. You're free to choose any `dim`, any feature extraction strategy,
and any backend as long as those invariants hold; if your backend is
network-bound, batch or cache internally, because `SemanticMemory` does
not offer an `embed_batch` hook today. A `SemanticMemory.embed_batch`
path and an approximate-NN index are not in the current roadmap —
[ROADMAP.md § Non-goals](https://github.com/jspwrd/mente/blob/main/ROADMAP.md#non-goals-at-least-for-now)
explicitly parks "production-grade vector DB" until a user asks.

## Add a synthesizer

The synthesizer turns computation-shaped intents into Python source that
the synthesis reasoner then sandboxes, validates, and promotes into the
verified-primitive library.

### Shape

From [`mente/synthesizers/__init__.py`](reference/synthesis.md):

```python
class Synthesizer(Protocol):
    def synthesize(
        self, intent_text: str
    ) -> tuple[str, str, dict[str, Any]] | None:
        """Return (source, entrypoint, args) or None if we can't synthesize."""
```

Two implementations ship: `TemplateSynthesizer` (deterministic, regex-
driven, zero-deps) and `LLMSynthesizer` (asks Claude for a pure function;
returns `None` on refusal / parse failure). `LLMSynthesizer` is lazily
imported so environments without the `anthropic` SDK can still use
`TemplateSynthesizer`.

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

### How existing defaults constrain your plug-in

`TemplateSynthesizer` emits pure functions with a stable `entrypoint`
name and JSON-serializable `args`. Your synthesizer MUST do the same:
`mente.synthesis._validate_ast` rejects the source if it contains any
of the disallowed node types (`Import`, `With`, `Lambda`, `Try`, …) or
disallowed names (`__import__`, `open`, `exec`, `eval`, `getattr`, …)
or any dunder attribute access, and the sandbox driver serializes
`args` through `json.dumps` before passing them to the entrypoint. Your
`synthesize` SHOULD return `None` rather than raising on routine
decline, so `SynthesisReasoner` can fall back to the next backend in a
chain. You're free to use any pattern-matching, LLM call, or grammar
you like to produce the source — the sandbox is the trust boundary,
not the synthesizer. Training `LLMSynthesizer` at scale is tracked in
[ROADMAP.md § Medium-term](https://github.com/jspwrd/mente/blob/main/ROADMAP.md#medium-term-03x--05x)
("LLM-authored synthesis at scale").

## Add a verifier

The verifier is consulted after a reasoner produces a response. It
returns an accept/reject verdict plus structured reasons. Rejections
can trigger rework or escalation.

### Shape

From [`mente/verifier.py`](reference/verifier.md); the canonical
`StructuredVerifier` Protocol lives in `mente.verifiers`:

```python
@dataclass
class Verdict:
    accept: bool
    score: float
    reasons: list[str]

class StructuredVerifier(Protocol):
    def verify(
        self, intent: Intent, response: Response, world: WorldModel
    ) -> Verdict: ...
```

`mente.verifier.Verifier` is a back-compat re-export of
`HeuristicVerifier` wrapped to log rejected verdicts at WARNING level.
`mente.verifiers.CompositeVerifier` lets you stack multiple verifiers
with a merge strategy; a trained step-verifier backend is tracked in
[ROADMAP.md § Medium-term](https://github.com/jspwrd/mente/blob/main/ROADMAP.md#medium-term-03x--05x).

### Minimal implementation

```python
from dataclasses import dataclass
from mente.verifiers import Verdict

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

### How existing defaults constrain your plug-in

`HeuristicVerifier` returns a `Verdict` with `score` clamped to
`[0.0, 1.0]`, an explicit `accept` flag, and a non-empty `reasons`
list (it adds `"ok"` when no other reason fires). `Runtime.handle_intent`
calls `self.verifier.verify(...)` synchronously on every turn and
persists `verdict.score` into the latent checkpoint. Your verifier
MUST return a `Verdict` (the dataclass, not a bool) and SHOULD populate
`reasons` with short machine-readable strings so the existing WARNING
log line stays useful. Your `verify` MUST be synchronous — the runtime
calls it inline on the event-loop turn. You're free to ignore
`intent`/`world` entirely, score however you like, and combine with
other verifiers via
`mente.verifiers.CompositeVerifier(verifiers=[...], strategy=...)`.

## Where to go next

- [Architecture](architecture.md) — why the surfaces are shaped this way
- [API reference](reference/index.md) — every public symbol with its signature
- [Build an agent](tutorials/build-an-agent.md) — a complete example
  that registers a tool and subscribes to intents
