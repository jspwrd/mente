# Contributing to ARIA

Thanks for your interest in ARIA. This is an early-stage project; small, focused
patches and discussion issues are both welcome.

## Getting set up

```bash
git clone https://github.com/jasperreed/aria.git
cd aria
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest -q          # unit tests
./aria test        # in-process smoke tests (bus, synthesis, memory)
```

## Linting and types

```bash
ruff check .
ruff format .
mypy src/aria
```

`pre-commit` is configured; enable it locally with `pre-commit install`.

## Submitting a PR

1. Fork and create a topic branch from `main`.
2. Keep commits focused; write a clear message explaining the *why*.
3. Make sure `pytest`, `./aria test`, and `ruff check .` pass.
4. Open a PR and fill out the template.

## Code style

- Python 3.11+ only. Use PEP 604 unions (`X | Y`), not `Optional`/`Union`.
- Prefer `@dataclass` for plain records and `typing.Protocol` for interfaces.
- Use `async`/`await` for I/O-bound code; keep CPU-bound code synchronous.
- Keep modules under `src/aria/` small and focused.
