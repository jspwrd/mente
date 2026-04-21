"""Tests for aria.synthesis — templated recognition, sandboxed execution, library persistence, AST gating."""
from __future__ import annotations

import pytest

from aria.synthesis import (
    LibraryStore,
    Primitive,
    SynthesisReasoner,
    TemplateSynthesizer,
    _validate_ast,
)
from aria.tools import ToolRegistry
from aria.types import Intent

from fixtures.cognition_helpers import make_world


# --- TemplateSynthesizer recognition ---------------------------------------


def test_template_recognizes_fib():
    out = TemplateSynthesizer().synthesize("compute fibonacci 10")
    assert out is not None
    src, entry, args = out
    assert entry == "fib"
    assert args == {"n": 10}
    assert "def fib" in src


def test_template_recognizes_factorial():
    out = TemplateSynthesizer().synthesize("factorial of 6")
    assert out is not None
    _src, entry, args = out
    assert entry == "factorial"
    assert args == {"n": 6}


def test_template_recognizes_power():
    out = TemplateSynthesizer().synthesize("compute 2 ** 10")
    assert out is not None
    _src, entry, args = out
    assert entry == "power"
    assert args == {"a": 2, "b": 10}


def test_template_returns_none_for_unrecognized():
    assert TemplateSynthesizer().synthesize("write me a poem") is None


# --- _validate_ast gate ----------------------------------------------------


def test_validate_rejects_import():
    with pytest.raises(ValueError, match="Import"):
        _validate_ast("import os\n")


def test_validate_rejects_from_import():
    with pytest.raises(ValueError, match="Import"):
        _validate_ast("from os import system\n")


def test_validate_rejects_dunder_import_call():
    with pytest.raises(ValueError, match="__import__"):
        _validate_ast("x = __import__('os')\n")


def test_validate_rejects_exec_name():
    with pytest.raises(ValueError, match="exec"):
        _validate_ast("exec('print(1)')\n")


def test_validate_rejects_dunder_attribute_access():
    with pytest.raises(ValueError, match="dunder"):
        _validate_ast("def f(x):\n    return x.__class__\n")


def test_validate_rejects_try_and_raise():
    with pytest.raises(ValueError):
        _validate_ast("def f():\n    try:\n        return 1\n    except: return 0\n")


def test_validate_accepts_pure_arithmetic_function():
    _validate_ast(
        "def fib(n):\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n):\n"
        "        a, b = b, a + b\n"
        "    return a\n"
    )


# --- SynthesisReasoner end-to-end (sandboxed subprocess) -------------------


async def test_synthesis_reasoner_computes_fib(tmp_path):
    library = LibraryStore(path=tmp_path / "lib.json")
    tools = ToolRegistry()
    r = SynthesisReasoner(library=library, tools=tools)
    world = await make_world()
    resp = await r.answer(Intent(text="compute fib 10"), world, ToolRegistry())
    assert "55" in resp.text
    assert resp.confidence >= 0.9
    # The primitive got promoted.
    assert len(library.list()) == 1


async def test_synthesis_reasoner_computes_factorial(tmp_path):
    library = LibraryStore(path=tmp_path / "lib.json")
    tools = ToolRegistry()
    r = SynthesisReasoner(library=library, tools=tools)
    world = await make_world()
    resp = await r.answer(Intent(text="factorial of 6"), world, ToolRegistry())
    assert "720" in resp.text
    assert resp.confidence >= 0.9


async def test_synthesis_reasoner_unrecognized_returns_zero_confidence(tmp_path):
    library = LibraryStore(path=tmp_path / "lib.json")
    tools = ToolRegistry()
    r = SynthesisReasoner(library=library, tools=tools)
    world = await make_world()
    resp = await r.answer(Intent(text="compose a haiku"), world, ToolRegistry())
    assert resp.text == ""
    assert resp.confidence == 0.0


async def test_synthesis_repeat_increments_invocations(tmp_path):
    library = LibraryStore(path=tmp_path / "lib.json")
    tools = ToolRegistry()
    r = SynthesisReasoner(library=library, tools=tools)
    world = await make_world()
    await r.answer(Intent(text="compute fib 5"), world, ToolRegistry())
    await r.answer(Intent(text="compute fib 5"), world, ToolRegistry())
    prims = library.list()
    assert len(prims) == 1
    assert prims[0].invocations >= 2


# --- LibraryStore persistence ---------------------------------------------


def test_library_persists_across_instantiation(tmp_path):
    path = tmp_path / "lib.json"
    store1 = LibraryStore(path=path)
    store1.add(Primitive(
        name="lib.fib.abc123",
        source="def fib(n):\n    return n\n",
        entrypoint="fib",
        signature={"n": "int"},
        invocations=1,
    ))
    # Reopen in a fresh instance: contents must survive.
    store2 = LibraryStore(path=path)
    assert store2.get("lib.fib.abc123") is not None
    assert store2.list()[0].invocations == 1


def test_library_save_is_readable_json(tmp_path):
    import json

    path = tmp_path / "lib.json"
    store = LibraryStore(path=path)
    store.add(Primitive(
        name="lib.demo",
        source="def demo():\n    return 1\n",
        entrypoint="demo",
        signature={},
        invocations=3,
    ))
    data = json.loads(path.read_text())
    assert "lib.demo" in data
    assert data["lib.demo"]["invocations"] == 3
