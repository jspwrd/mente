"""Tests for aria.specialists.MathSpecialist — arithmetic pattern recognition + safe AST eval."""
from __future__ import annotations

from aria.specialists import MathSpecialist, _safe_eval
from aria.tools import ToolRegistry
from aria.types import Intent

import ast

import pytest

from fixtures.cognition_helpers import make_world


async def test_math_addition():
    r = MathSpecialist()
    world = await make_world()
    resp = await r.answer(Intent(text="what is 2 + 3?"), world, ToolRegistry())
    assert "= 5" in resp.text
    assert resp.confidence >= 0.95
    assert resp.tier == "specialist"


async def test_math_multiplication_and_precedence():
    r = MathSpecialist()
    world = await make_world()
    resp = await r.answer(Intent(text="compute 2 + 3 * 4"), world, ToolRegistry())
    assert "= 14" in resp.text


async def test_math_parentheses():
    r = MathSpecialist()
    world = await make_world()
    resp = await r.answer(Intent(text="evaluate (1 + 2) * 3"), world, ToolRegistry())
    assert "= 9" in resp.text


async def test_math_unary_minus():
    r = MathSpecialist()
    world = await make_world()
    resp = await r.answer(Intent(text="calculate -5 + 10"), world, ToolRegistry())
    assert "= 5" in resp.text


async def test_math_division_yields_float():
    r = MathSpecialist()
    world = await make_world()
    resp = await r.answer(Intent(text="compute 7 / 2"), world, ToolRegistry())
    assert "3.5" in resp.text


async def test_math_non_arithmetic_intent_zero_confidence():
    r = MathSpecialist()
    world = await make_world()
    resp = await r.answer(Intent(text="tell me a joke"), world, ToolRegistry())
    assert resp.confidence == 0.0
    assert resp.text == ""


async def test_math_without_trigger_verb_zero_confidence():
    """Plain arithmetic without a recognition verb should not be picked up."""
    r = MathSpecialist()
    world = await make_world()
    resp = await r.answer(Intent(text="2 + 3"), world, ToolRegistry())
    assert resp.confidence == 0.0


async def test_math_unparseable_expression_signals_failure():
    r = MathSpecialist()
    world = await make_world()
    # Matches the recognition verb but has a trailing operator — parser error.
    resp = await r.answer(Intent(text="what is 2 + +"), world, ToolRegistry())
    assert resp.confidence == pytest.approx(0.1)
    assert "could not evaluate" in resp.text


def test_safe_eval_rejects_name_reference():
    tree = ast.parse("x + 1", mode="eval")
    with pytest.raises(ValueError):
        _safe_eval(tree)


def test_safe_eval_rejects_function_call():
    tree = ast.parse("len([1,2,3])", mode="eval")
    with pytest.raises(ValueError):
        _safe_eval(tree)


def test_safe_eval_accepts_nested_binop():
    tree = ast.parse("(1 + 2) * (3 - 4)", mode="eval")
    assert _safe_eval(tree) == -3
