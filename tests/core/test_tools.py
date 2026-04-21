"""Tests for aria.tools.ToolRegistry: register, signature, invoke, ToolResult."""
from __future__ import annotations

import asyncio

import pytest

from aria.tools import ToolRegistry, ToolResult, ToolSpec


@pytest.mark.asyncio
async def test_register_extracts_signature():
    reg = ToolRegistry()

    @reg.register(name="add", description="add two ints", returns="int")
    async def add(a: int, b: int) -> int:
        return a + b

    spec = reg.get("add")
    assert isinstance(spec, ToolSpec)
    assert spec.name == "add"
    assert spec.description == "add two ints"
    assert spec.returns == "int"
    # params carries type-hint string form for each arg.
    assert set(spec.params.keys()) == {"a", "b"}


@pytest.mark.asyncio
async def test_register_default_est_cost_ms():
    reg = ToolRegistry()

    @reg.register(name="noop", description="", returns="None")
    async def noop() -> None:
        return None

    spec = reg.get("noop")
    assert spec is not None
    assert spec.est_cost_ms == 5.0


@pytest.mark.asyncio
async def test_register_custom_est_cost_ms():
    reg = ToolRegistry()

    @reg.register(name="slow", description="", returns="None", est_cost_ms=250.0)
    async def slow() -> None:
        return None

    assert reg.get("slow").est_cost_ms == 250.0


@pytest.mark.asyncio
async def test_register_handles_unannotated_params():
    reg = ToolRegistry()

    @reg.register(name="t", description="", returns="Any")
    async def t(x, y=2):  # no annotations
        return x + y

    spec = reg.get("t")
    assert spec.params["x"] == "Any"
    assert spec.params["y"] == "Any"


@pytest.mark.asyncio
async def test_invoke_returns_tool_result_on_success():
    reg = ToolRegistry()

    @reg.register(name="add", description="", returns="int")
    async def add(a: int, b: int) -> int:
        return a + b

    res = await reg.invoke("add", a=2, b=3)
    assert isinstance(res, ToolResult)
    assert res.tool == "add"
    assert res.ok is True
    assert res.value == 5
    assert res.error is None
    assert res.cost_ms >= 0.0


@pytest.mark.asyncio
async def test_invoke_unknown_tool_returns_error():
    reg = ToolRegistry()
    res = await reg.invoke("does_not_exist", foo=1)
    assert res.ok is False
    assert res.value is None
    assert res.error is not None
    assert "unknown tool" in res.error
    assert res.cost_ms == 0.0


@pytest.mark.asyncio
async def test_invoke_captures_exception_as_error():
    reg = ToolRegistry()

    @reg.register(name="boom", description="", returns="None")
    async def boom() -> None:
        raise ValueError("nope")

    res = await reg.invoke("boom")
    assert res.ok is False
    assert res.value is None
    assert res.error is not None
    assert "ValueError" in res.error or "nope" in res.error
    assert res.cost_ms >= 0.0


@pytest.mark.asyncio
async def test_invoke_measures_cost_ms():
    reg = ToolRegistry()

    @reg.register(name="wait", description="", returns="None")
    async def wait() -> None:
        await asyncio.sleep(0.02)

    res = await reg.invoke("wait")
    assert res.ok is True
    # At least ~20 ms; allow generous slack but it must be measurably > 0.
    assert res.cost_ms >= 15.0


@pytest.mark.asyncio
async def test_list_returns_registered_tools():
    reg = ToolRegistry()

    @reg.register(name="a", description="", returns="int")
    async def a() -> int:
        return 1

    @reg.register(name="b", description="", returns="int")
    async def b() -> int:
        return 2

    names = {s.name for s in reg.list()}
    assert names == {"a", "b"}


@pytest.mark.asyncio
async def test_get_missing_returns_none():
    reg = ToolRegistry()
    assert reg.get("nope") is None


@pytest.mark.asyncio
async def test_register_overwrites_same_name():
    reg = ToolRegistry()

    @reg.register(name="dup", description="v1", returns="int")
    async def v1() -> int:
        return 1

    @reg.register(name="dup", description="v2", returns="int")
    async def v2() -> int:
        return 2

    res = await reg.invoke("dup")
    assert res.value == 2
    assert reg.get("dup").description == "v2"


@pytest.mark.asyncio
async def test_invoke_passes_kwargs_through():
    reg = ToolRegistry()
    seen = {}

    @reg.register(name="capture", description="", returns="None")
    async def capture(x: int, y: str) -> None:
        seen["x"] = x
        seen["y"] = y

    res = await reg.invoke("capture", x=7, y="hi")
    assert res.ok is True
    assert seen == {"x": 7, "y": "hi"}
