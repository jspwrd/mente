"""Guard against syntax errors / broken imports in examples/*.py.

Loads each example module via `importlib` without running its `main()`.
This is a cheap smoke test for the example gallery: the scripts are
meant to be self-contained, so a typo or a stale import shouldn't
slip in undetected.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Project root: parent of this file's parent (tests/ -> project root).
_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES = _ROOT / "examples"


# Ensure the in-tree package is importable for the example modules.
sys.path.insert(0, str(_ROOT / "src"))


@pytest.mark.parametrize(
    "script_name",
    ["coding_agent.py", "research_agent.py", "personal_org.py"],
)
def test_example_importable(script_name: str) -> None:
    path = _EXAMPLES / script_name
    assert path.exists(), f"missing example: {path}"

    spec = importlib.util.spec_from_file_location(
        f"mente_examples.{path.stem}", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Loading the module executes top-level code (imports, constants) but
    # not `main()` — the `if __name__ == "__main__"` guard blocks that.
    spec.loader.exec_module(module)

    # Each example should expose an async `main` entry point.
    assert hasattr(module, "main"), f"{script_name} has no main()"
