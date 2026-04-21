# Changelog

All notable changes to this project follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Project renamed from `aria` to `mente`. All modules, imports, CLI commands,
  env vars, and data directories are now under the `mente` namespace.

## [0.1.0] — Unreleased (initial public release)

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

[Unreleased]: https://github.com/jspwrd/mente/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jspwrd/mente/releases/tag/v0.1.0
