# Contributors

People who have shaped mente.

## Authors

- **Jasper Reed** ([@jspwrd](https://github.com/jspwrd)) — creator and
  primary maintainer.

## Contributors

_Your name here._ See [CONTRIBUTING.md](CONTRIBUTING.md) for the PR flow.
This file is maintained by hand; we also automatically list contributors on
[GitHub](https://github.com/jspwrd/mente/graphs/contributors).

## Acknowledgements

mente's architectural shape is indebted to a lot of prior work. Some of the
ideas that most directly influenced this design:

- **Claude Code** (Anthropic) — the "harness around one model" pattern this
  framework is deliberately *not*.
- **LangGraph / AutoGen / CrewAI** — they taught me what happens when an
  agent framework grows too many abstractions.
- **Dreamer / MuZero / RETRO** — the "latent-state rollouts + differentiable
  memory" ideas in the architecture essay come from this lineage.
- **Active Inference** (Karl Friston et al.) — the intrinsic-objective /
  free-energy framing behind the curiosity loop.
- **Zettelkasten / `Memorizing Transformer`** — the tiered-memory design.
- The **Python ecosystem** — `asyncio`, `sqlite3`, `hatchling`, `uv`, `ruff`,
  `mkdocs-material`, `pytest`. Every one of those is someone's long work.

## How to get listed

Open a PR — any accepted change lands your name here (and on the GitHub
contributors graph automatically). Non-code contributions count too: docs,
examples, bug reports with reproductions, design feedback.
