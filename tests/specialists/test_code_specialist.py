"""Unit tests for CodeSpecialist.

The specialist is intentionally a shallow, stdlib-only critic. These tests
lock in the kinds of signals it should surface today; deeper analysis is
left for Phase 2.
"""
from __future__ import annotations

import asyncio
import textwrap

import pytest

from mente.specialists import CodeSpecialist, MathSpecialist
from mente.types import Intent


def _ask(code_specialist, world_tools, text):
    world, tools = world_tools
    return asyncio.run(code_specialist.answer(Intent(text=text), world, tools))


def _fence(code):
    return f"review this code:\n```python\n{textwrap.dedent(code).strip()}\n```"


@pytest.fixture
def cs():
    return CodeSpecialist()


# --- contract / shape -------------------------------------------------------

def test_reexport_from_specialists_module():
    # Critical invariant: peer.py etc. do `from mente.specialists import MathSpecialist`.
    assert MathSpecialist().name == "specialist.math"
    assert CodeSpecialist().name == "specialist.code"


def test_tier_and_cost(cs):
    assert cs.tier == "specialist"
    assert cs.est_cost_ms == 25.0


# --- negative: not code -----------------------------------------------------

def test_non_code_input_zero_confidence(cs, world_tools):
    r = _ask(cs, world_tools, "tell me a joke about pelicans")
    assert r.confidence == 0.0
    assert r.text == ""


def test_empty_code_block_zero_confidence(cs, world_tools):
    r = _ask(cs, world_tools, "review this code:\n```\n\n```")
    assert r.confidence == 0.0


# --- syntax errors ---------------------------------------------------------

def test_syntax_error_reported(cs, world_tools):
    r = _ask(cs, world_tools, _fence("def f(:\n    pass"))
    assert "parse error" in r.text.lower() or "syntax" in r.text.lower()
    assert r.confidence == 0.95


# --- mutable default args --------------------------------------------------

def test_mutable_default_list(cs, world_tools):
    r = _ask(cs, world_tools, _fence("""
        def f(x=[]) -> list:
            return x
    """))
    assert "mutable default" in r.text


def test_mutable_default_dict(cs, world_tools):
    r = _ask(cs, world_tools, _fence("""
        def f(x={}) -> dict:
            return x
    """))
    assert "mutable default" in r.text


# --- bare except -----------------------------------------------------------

def test_bare_except(cs, world_tools):
    r = _ask(cs, world_tools, _fence("""
        def f() -> None:
            try:
                do_thing()
            except:
                pass
    """))
    assert "bare `except" in r.text or "bare except" in r.text.lower()


# --- unused imports --------------------------------------------------------

def test_unused_import(cs, world_tools):
    r = _ask(cs, world_tools, _fence("""
        import os
        import sys

        def main() -> None:
            print(sys.argv)
    """))
    assert "unused import `os`" in r.text
    assert "unused import `sys`" not in r.text


def test_used_import_not_flagged(cs, world_tools):
    r = _ask(cs, world_tools, _fence("""
        import json

        def dump(x) -> str:
            return json.dumps(x)
    """))
    assert "unused import" not in r.text


# --- undefined names -------------------------------------------------------

def test_undefined_name(cs, world_tools):
    r = _ask(cs, world_tools, _fence("""
        def g(x):
            return mystery_func(x)
    """))
    assert "undefined name `mystery_func`" in r.text


def test_builtin_not_flagged_as_undefined(cs, world_tools):
    r = _ask(cs, world_tools, _fence("""
        def h(xs) -> int:
            return len(xs)
    """))
    assert "undefined name" not in r.text


def test_parameters_not_flagged_as_undefined(cs, world_tools):
    r = _ask(cs, world_tools, _fence("""
        def add(a: int, b: int) -> int:
            return a + b
    """))
    assert "undefined name" not in r.text


# --- missing return hints --------------------------------------------------

def test_missing_return_hint(cs, world_tools):
    r = _ask(cs, world_tools, _fence("""
        def bump(x):
            return x + 1
    """))
    assert "missing a return type hint" in r.text


def test_present_return_hint_ok(cs, world_tools):
    r = _ask(cs, world_tools, _fence("""
        def bump(x: int) -> int:
            return x + 1
    """))
    assert "missing a return type hint" not in r.text


def test_init_return_hint_not_flagged(cs, world_tools):
    r = _ask(cs, world_tools, _fence("""
        class Foo:
            def __init__(self, x):
                self.x = x
    """))
    assert "missing a return type hint" not in r.text


# --- print leftover --------------------------------------------------------

def test_print_call_flagged(cs, world_tools):
    r = _ask(cs, world_tools, _fence("""
        def f(x: int) -> int:
            print(x)
            return x
    """))
    assert "print(" in r.text and "debug" in r.text


# --- TODO comments ---------------------------------------------------------

def test_todo_comment(cs, world_tools):
    r = _ask(cs, world_tools, _fence("""
        def f() -> None:
            # TODO: handle edge case
            return
    """))
    assert "TODO comment" in r.text


def test_fixme_comment(cs, world_tools):
    r = _ask(cs, world_tools, _fence("""
        def f() -> None:
            # FIXME: wrong constant below
            x: int = 1
            return
    """))
    assert "FIXME comment" in r.text


# --- clean code ------------------------------------------------------------

def test_clean_code_no_findings(cs, world_tools):
    r = _ask(cs, world_tools, _fence("""
        from typing import Iterable

        def total(xs: Iterable[int]) -> int:
            acc: int = 0
            for x in xs:
                acc += x
            return acc
    """))
    assert r.confidence == 0.9
    assert "no obvious issues" in r.text


# --- intent-shape detection ------------------------------------------------

def test_trigger_phrase_without_fence(cs, world_tools):
    r = _ask(cs, world_tools, "what's wrong with\ndef f(x=[]) -> list:\n    return x")
    # should pick up the mutable default even without a triple-backtick fence
    assert "mutable default" in r.text


def test_bare_code_with_def_recognized(cs, world_tools):
    # no trigger phrase, no fence, but has `def` tokens — we should still review.
    r = _ask(cs, world_tools, "def noop(x):\n    return x\n")
    assert r.confidence >= 0.9
