# Contributing to mente

Thanks for your interest. This is an early-stage project — small, focused
patches and design discussions are equally welcome. The goal is a framework
that stays readable in one sitting, so we're deliberately cautious about
complexity and abstractions.

## Getting set up

```bash
git clone https://github.com/jasperreed/mente.git
cd mente
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest -q          # unit tests
./mente test        # in-process smoke tests (bus, synthesis, memory)
```

## Linting and types

```bash
ruff check .
ruff format .
mypy src/mente
```

`pre-commit` is configured; enable it locally with `pre-commit install`.

## Submitting a PR

1. Fork and create a topic branch from `main`.
2. Keep commits focused; write a clear message explaining the *why*.
3. Make sure `pytest`, `./mente test`, and `ruff check .` pass.
4. Open a PR and fill out the template.

## Code style

- Python 3.11+ only. Use PEP 604 unions (`X | Y`), not `Optional`/`Union`.
- Prefer `@dataclass` for plain records and `typing.Protocol` for interfaces.
- Use `async`/`await` for I/O-bound code; keep CPU-bound code synchronous.
- Keep modules under `src/mente/` small and focused — aim for <200 lines per file.
- Module-top docstrings should note what's Phase 1 (shipping) vs Phase 2 (planned).

## What we love in a PR

- A failing test that demonstrates the bug, then the fix.
- New features guarded behind an optional dep group (don't add a runtime
  dependency to the core package).
- A docs/example update if the change affects the public API.
- Short, descriptive commit messages — "fix X" beats "improvements".

## What we're careful about

- Adding abstractions for speculative future needs — we'd rather have three
  similar lines than a premature base class.
- Adding runtime dependencies to the core package.
- Large refactors without prior discussion — open an issue first.

## Security

Security issues: please follow [SECURITY.md](SECURITY.md) — do **not** open
a public issue.
