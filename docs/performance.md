# Performance notes

MENTE is small enough that micro-benchmarks matter less than the shape of the
system. Still, it helps to know what's fast, what's slow, and what becomes a
bottleneck as you push it. The numbers below are rough — label them in your
head as "order of magnitude", not "SLA".

## Rough numbers

Per-intent latency, warm cache, no network:

| Tier | Typical latency | Notes |
|---|---|---|
| Fast heuristic | < 5 ms | Pattern match + tool call; dominates mundane traffic. |
| `MathSpecialist` | < 10 ms | Pure Python arithmetic, no synthesis. |
| `SynthesisReasoner` | 200–400 ms first call · < 30 ms cached | First call compiles and verifies a primitive; subsequent calls hit the library. |
| `CodeSpecialist` | 10–50 ms | Scales with snippet size and AST traversal depth. |
| `DeepSimulatedReasoner` (stub) | ~400 ms | Intentional artificial latency so the offline path feels like a slow tier. |
| `AnthropicReasoner` (Claude Opus 4.7, adaptive thinking) | 2–6 s | Dominated by the API call; prompt caching trims repeat prefixes. |

### Measurement methodology

These numbers come from timing the full `Runtime.handle_intent` path,
single-threaded Python 3.13 on an M-series Mac, localhost, with warm caches
(the library, semantic embeddings, and prompt cache already populated). Cold
first-run numbers are 2–3× higher for the tiers that load SQLite or touch the
embedding cache. Wall-clock on Linux x86 tends to be in the same ballpark;
Windows has not been profiled.

We did not benchmark under load, memory pressure, or with the federated TCP
bus hot-looping — treat anything outside the local single-process case as an
open question.

## Scaling limits

- **Semantic memory is a linear scan.** `SemanticMemory.search` computes
  cosine similarity against every row in `semantic.sqlite`. It's fine up to
  ~100K rows; past that, expect recall latency to dominate and swap in a real
  vector index (FAISS, pgvector, LanceDB) behind the `Embedder` protocol.
- **The TCP bus is local-federation shaped.** It's ergonomic for multi-process
  setups on one host but was not designed for WAN links — no auth, no
  compression, no backpressure past the asyncio buffer. For cross-host
  deployments, put it behind a VPN or rewrite the transport.
- **The consolidator grows with episodic log size.** It runs every
  `consolidator_interval_s` (default 10 s) and reads the recent tail of
  `episodic.sqlite`. If you let the log grow unbounded, each consolidation
  walks more rows; truncate periodically or tune the interval up.

## Tuning knobs

Common goals and which `MenteConfig` fields to touch:

| Goal | Fields | Direction |
|---|---|---|
| Lower latency | `consolidator_interval_s`, `curiosity_interval_s`, `router_ms_per_conf` | Raise the first two to reduce background work; lower the third to bias the router away from deep-tier escalation. |
| Lower LLM cost | `llm_effort`, `llm_max_tokens`, `router_min_confidence` | Drop effort to `"low"`, cap tokens, and raise the router's confidence floor so cheap tiers win more often. |
| Smaller memory footprint | `data_root`, `consolidator_interval_s` | Put `.mente/` on a scratch disk; consolidate more aggressively so episodic rows get digested and can be pruned. |
| Stricter answers | `verifier_min_confidence` | Raise it — more answers get rejected, fewer weak results leak through. |

Everything lives in `src/mente/config.py`; set values via TOML, env vars
(`MENTE_<FIELD_UPPER>`), or directly on `MenteConfig`.
