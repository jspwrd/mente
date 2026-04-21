# Roadmap

A loosely-ordered list of what's next. Contributions welcome — pick
anything that looks interesting and open a PR (or an issue to discuss first
if the scope is large).

## Type-check ratchet

`mypy` is advisory in CI today. Tighten module-by-module:

- [ ] `types.py`, `state.py`, `bus.py`, `memory.py` → `disallow_untyped_defs`
- [ ] `world_model.py`, `tools.py`, `transport.py`, `verifier.py` → same
- [ ] `reasoners.py`, `router.py`, `metacog.py` → same
- [ ] `specialists/`, `synthesizers/`, `embedders/`, `verifiers/` → same
- [ ] `runtime.py`, `cli.py`, `discovery.py` → same (harder, generic-heavy)
- [ ] Final: flip `strict = true` repo-wide and drop `continue-on-error`.

## Near-term (0.2.x)

- **Secrets redaction in logs** — apply `mente.logging.redact_secrets`
  to the structured logger by default.
- **Demo GIF in README** — record `./mente run` with `asciinema` / `vhs`.
- **Launch post** — 800-word technical essay for HN/lobste.rs.
- **Auto-update README section on `./mente` output** — a small doctest
  that re-renders the sample REPL session so it never drifts.

## Medium-term (0.3.x – 0.5.x)

- **Trained metacog head** — replace the pattern-coded estimator with a
  learned model that predicts confidence/cost per reasoner from intent
  features.
- **Trained step verifier** — process-reward-model-style scorer that
  complements the heuristic one.
- **LLM-authored synthesis at scale** — test harness for
  `LLMSynthesizer`, library curation, test-driven acceptance before
  promotion.
- **NATS transport** — drop-in for `TCPTransport`; enables multi-machine
  federation without a hub node.
- **Authenticated discovery** — signed capability manifests, peer health
  scoring, load balancing.
- **Observability node** — structured bus tap + Loki/Grafana recipe.
- **Dream consolidation** — LLM-backed `Consolidator` that writes
  natural-language digests and promotes stable beliefs to the world model.

## Long-term / research (0.x → 1.0)

- **Curiosity objective beyond heuristics** — free-energy / empowerment /
  information-gain-driven self-prompting.
- **World model as a predictor** — differentiable model that predicts
  next-state given action; enables imagined-rollout planning.
- **Intersubjective latent protocol** — co-evolved latent communication
  between peers (replace JSON-over-TCP with a learned compression).
- **Modular alignment** — separate the alignment layer from capability
  weights so updates don't trade off.

## Non-goals (at least for now)

- **Windows support** — macOS / Linux only.
- **Production-grade vector DB** — SQLite + cosine handles 100K entries.
  Swap only when a user asks.
- **Plugin marketplace / extension registry** — don't build this before
  there are users asking for it.

## How to influence this list

Open an issue with the tag `roadmap` if you want to reshuffle priorities.
Concrete use cases beat abstract wishes.
