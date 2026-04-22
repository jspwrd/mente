# Changelog

All notable changes to this project follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] — 2026-04-21

Post-launch hardening release. Every module now logs, every load-bearing
module type-checks under `mypy --strict`, test count jumped from 388 to 430,
and every audited rough edge is either fixed, covered by a test, or
explicitly documented. No breaking changes to the public API.

### Added
- **`LocalEmbedder`** (`mente.embedders.LocalEmbedder`) behind the new
  `mente[embeddings-local]` extra — CPU-friendly sentence-transformers path
  for offline users who want real semantic similarity without an API key.
- **`install.sh` one-liner installer** — `curl | bash` bootstrap that picks
  `uv tool install` → `pipx` → `pip --user` in fallback order, with
  `--with=<extras>`, `--version=`, `--pre`, `--dry-run` flags.
- **TCP transport authentication** — optional `auth_secret` (HMAC-SHA256
  handshake with 60s skew window, 4KB line cap). Unset on either side ⇒
  skipped, preserving the default inproc/demo flow. Configurable via
  `MENTE_BUS_SECRET`.
- **`RemoteReasoner` backpressure** — `_pending` futures now cleared on
  every exit path; `_prune_stale()` drops futures older than
  `2 * timeout_s` so a disappearing peer cannot leak memory.
- **Synthesis sandbox hardening** — POSIX `RLIMIT_CPU=2s`,
  `RLIMIT_AS=256MB`, `RLIMIT_NOFILE=32`, `RLIMIT_NPROC=0` via `preexec_fn`;
  minimal `env={"PATH": "/usr/bin:/bin"}`; explicit `close_fds=True`;
  top-of-module threat model documenting what the sandbox does and doesn't
  defend against.
- **`CircuitBreaker` wrapped around Claude calls** — per-instance
  (threshold=5, recovery=60s). When open, `AnthropicReasoner.answer()`
  returns a `confidence=0.0` response immediately instead of hitting the
  API.
- **`MenteConfig` adoption across remaining modules** —
  `AnthropicReasoner`, `Curiosity`, `Consolidator` now accept an optional
  `config: MenteConfig | None` and read their tunables from it. CLI loads
  env config via `MenteConfig.load()`.
- **Structured logging in bus / router / verifier / llm / discovery /
  transport / consolidator** — `mente.logging.get_logger(...)` calls
  throughout, with `extra={"trace_id": ...}` propagation. Router emits
  INFO on dispatch + escalation; verifier WARNs on rejection; bus DEBUGs
  on subscribe/publish.
- **Secrets redaction filter** installed on the default log handler —
  API-key-like substrings masked before emission.
- **`CodeSpecialist` registered in the default roster** plus
  `_SPECIALIST_PATS` entry in metacog, so code-shaped intents route to it
  automatically.
- **`self_model.answer()` dispatch table** replacing the if/elif chain.
- **PEP 561 `py.typed` marker** shipped in the wheel + CI `wheel-install`
  job that verifies packaging integrity.
- **Sigstore attestations** on PyPI/TestPyPI publishes (`attestations:
  true` on `pypa/gh-action-pypi-publish`).
- **mkdocs `--strict`** in the docs CI.
- **4 new test files** — federated CLI e2e, runtime shutdown under error,
  synthesis library reload cycle, LocalEmbedder via mocked ST.
- **New docs** — `docs/performance.md` (rough numbers + tuning knobs),
  `docs/faq.md` (6 questions), no-install path in quickstart, Phase-2-made-
  concrete pass on `docs/extending.md`, richer Google-style docstrings on
  `Runtime`, `EventBus`, `SlowMemory`, `FastHeuristicReasoner`.
- **Full Protocol docstrings** on `Reasoner`, `Embedder`, `Synthesizer`,
  `StructuredVerifier` with drop-in implementation snippets.

### Changed
- **Project renamed `aria` → `mente`** across all modules, imports, CLI
  commands, `MENTE_*` env var prefix, `.mente/` data dir, `AriaConfig` →
  `MenteConfig`. One-time migration; no ongoing back-compat shim.
- **`Runtime.__post_init__`** split into six phased private methods
  (`_setup_logging_and_storage` → `_setup_reasoners` → `_setup_router` →
  `_setup_tools_and_subscribers` → `_setup_self_model` →
  `_setup_background_surfaces`). Behavior-preserving.
- **`Runtime.shutdown()` is now idempotent** — double-call is a no-op via
  `_shutdown_done` flag.
- **`HeuristicVerifier` checks expanded** — numeric range sanity, tool-
  call corroboration, repetition detection, hallucinated-URL flag, tier-
  aware thresholds.
- **mypy strict on 13 modules**: `types`, `state`, `self_model`, `runtime`,
  `synthesis`, `transport`, `discovery`, `curiosity`, `consolidator`,
  `llm`, `bus`, `router`, `verifier`.

### Fixed
- **Curiosity idle-threshold test** replaced real-clock `asyncio.sleep`
  with a deterministic sentinel check — no more CI flakes.
- **Ruff import-sorting + style** cleaned up across the tree;
  `contextlib.suppress` used where appropriate; `zip(..., strict=True)`
  where length mismatches are a bug.

## [0.1.0] — 2026-04-21 (initial public release)

### Added
- **Core event-driven runtime** (`mente.runtime.Runtime`): async event loop,
  persistent latent state, world model, typed tools, pluggable transport.
- **Heterogeneous reasoner tiers** with `Reasoner` Protocol:
  `FastHeuristicReasoner`, `DeepSimulatedReasoner`, `AnthropicReasoner`
  (Claude Opus 4.7 with adaptive thinking + prompt caching, behind
  `mente[llm]` optional dep).
- **Metacog + router** that dispatch intents by predicted confidence/cost,
  with automatic escalation on low-confidence responses.
- **Step-wise verifier** (`mente.verifiers`): `HeuristicVerifier` with
  numeric sanity, tool-call corroboration, repetition detection, world-
  model contradiction checks; `CompositeVerifier` chain.
- **Tiered memory**: `FastMemory` (TTL), `SlowMemory` (SQLite episodic log
  with `summarize()`), `SemanticMemory` (cosine search over vectors).
- **Pluggable embedders** (`mente.embedders`): `HashEmbedder` (character
  n-gram, offline); `VoyageEmbedder` (behind `mente[embeddings]`).
- **Program synthesis + verified primitive library** (`mente.synthesis`,
  `mente.synthesizers`): `TemplateSynthesizer` (regex-based, offline),
  `LLMSynthesizer` (Claude-authored, behind `mente[llm]`). Both route
  through AST validation + subprocess sandbox; verified snippets are
  promoted to persistent tools in `LibraryStore`.
- **Specialists** (`mente.specialists`): `MathSpecialist` (safe arithmetic),
  `CodeSpecialist` (AST-based static analysis: undefined names, unused
  imports, mutable defaults, bare except, missing hints, TODO detection).
- **Distributed bus** (`mente.bus`, `mente.transport`): pluggable transport
  with in-process and TCP (hub/spoke) implementations.
- **Federation** (`mente.discovery`): capability announcements, peer
  directory, `RemoteReasoner` proxy, `RemoteRequestHandler`.
- **Background cognition**: `Consolidator` ("sleep cycle" that distills
  episodes into digests), `Curiosity` (idle-time self-prompting from
  world-model gaps).
- **Self-model** (`mente.self_model`): structured, queryable self-
  representation covering reasoners, tools, latent state, recent digest.
- **Framework primitives**: `mente.config.MenteConfig` (TOML + env
  override), `mente.logging` (structured logging helpers with JSON
  formatter, context binding, secret redaction), `mente.resilience`
  (`retry_async`, `timeout`, `CircuitBreaker`).
- **CLI** (`mente.cli`): unified entry point with `run` (interactive REPL),
  `demo` (scripted walkthrough), `federated` (single-process hub + peer
  with TCP bus between them), `peer`, `test`, `reset`.
- **Example gallery**: `coding_agent.py`, `research_agent.py`,
  `personal_org.py` — complete runnable examples.
- **Documentation**: architecture deep-dive, auto-generated API reference
  (mkdocs-material + mkdocstrings), quickstart tutorial, build-an-agent
  tutorial, extension guide.
- **Tests**: 388 passing (core, cognition, advanced, consolidator,
  synthesizers, verifiers, embedders, specialists, config, logging,
  resilience, examples).
- **CI** on Python 3.11 / 3.12 / 3.13 × ubuntu / macos (ruff, mypy, pytest,
  `./mente test`).
- **Packaging**: MIT license, classifiers, optional dep groups (`llm`,
  `embeddings`, `docs`, `dev`), pre-commit config.

### Known limitations
- Hash embedder is a toy; install `mente[embeddings]` and use `VoyageEmbedder`
  for real semantic similarity.
- Default deep reasoner is a stub; install `mente[llm]` and set
  `ANTHROPIC_API_KEY` for real LLM responses.
- Discovery protocol is unauthenticated; safe for single-user /
  trusted-LAN setups only.
- No Windows support (macOS / Linux only).

[Unreleased]: https://github.com/jspwrd/mente/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/jspwrd/mente/releases/tag/v0.2.0
[0.1.0]: https://github.com/jspwrd/mente/releases/tag/v0.1.0
