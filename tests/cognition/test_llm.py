"""Tests for mente.llm.AnthropicReasoner — constructor gating, prompt structure,
model/thinking params, error path. Never calls the real API.
"""
from __future__ import annotations

import os

import pytest

from mente import llm
from mente.tools import ToolRegistry
from mente.types import Belief, Intent

from fixtures.cognition_helpers import make_world
from fixtures.fake_llm import FakeAsyncAnthropic, make_fake_anthropic_module


def _install_fake(monkeypatch, **fake_kwargs):
    """Patch mente.llm so AnthropicReasoner.__post_init__ uses a fake SDK."""
    module, client = make_fake_anthropic_module(**fake_kwargs)
    monkeypatch.setattr(llm, "anthropic", module, raising=False)
    monkeypatch.setattr(llm, "_ANTHROPIC_AVAILABLE", True, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return client


def test_constructor_requires_api_key(monkeypatch):
    module, _ = make_fake_anthropic_module()
    monkeypatch.setattr(llm, "anthropic", module, raising=False)
    monkeypatch.setattr(llm, "_ANTHROPIC_AVAILABLE", True, raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        llm.AnthropicReasoner()


def test_constructor_requires_sdk(monkeypatch):
    monkeypatch.setattr(llm, "_ANTHROPIC_AVAILABLE", False, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with pytest.raises(RuntimeError, match="anthropic SDK"):
        llm.AnthropicReasoner()


def test_anthropic_available_reflects_env(monkeypatch):
    monkeypatch.setattr(llm, "_ANTHROPIC_AVAILABLE", True, raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert llm.anthropic_available() is False

    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert llm.anthropic_available() is True

    monkeypatch.setattr(llm, "_ANTHROPIC_AVAILABLE", False, raising=False)
    assert llm.anthropic_available() is False


async def test_default_model_is_claude_opus_4_7(monkeypatch):
    client = _install_fake(monkeypatch)
    r = llm.AnthropicReasoner()
    world = await make_world()
    await r.answer(Intent(text="hi"), world, ToolRegistry())
    assert client.calls, "expected a messages.create call"
    call = client.calls[0]
    assert call["model"] == "claude-opus-4-7"


async def test_thinking_parameter_is_adaptive(monkeypatch):
    client = _install_fake(monkeypatch)
    r = llm.AnthropicReasoner()
    world = await make_world()
    await r.answer(Intent(text="hi"), world, ToolRegistry())
    call = client.calls[0]
    assert call["thinking"] == {"type": "adaptive"}


async def test_system_prompt_has_cache_breakpoint_on_preamble(monkeypatch):
    client = _install_fake(monkeypatch)
    r = llm.AnthropicReasoner()
    world = await make_world()
    await r.answer(Intent(text="hi"), world, ToolRegistry())
    system = client.calls[0]["system"]
    assert isinstance(system, list) and len(system) >= 2
    preamble, context = system[0], system[1]
    assert preamble["type"] == "text"
    assert preamble.get("cache_control") == {"type": "ephemeral"}
    # Preamble must be the FROZEN MENTE instructions (not the volatile snapshot).
    assert "MENTE" in preamble["text"]
    # Context block must NOT be cache-controlled — it changes turn to turn.
    assert "cache_control" not in context
    assert context["type"] == "text"


async def test_context_block_renders_world_model_after_breakpoint(monkeypatch):
    client = _install_fake(monkeypatch)
    r = llm.AnthropicReasoner()
    world = await make_world([Belief(entity="user", attribute="name", value="Ada")])
    await r.answer(Intent(text="who am I"), world, ToolRegistry())
    system = client.calls[0]["system"]
    context_text = system[1]["text"]
    assert "WORLD MODEL" in context_text
    assert "user.name" in context_text
    assert "Ada" in context_text


async def test_context_block_lists_available_tools(monkeypatch):
    client = _install_fake(monkeypatch)
    r = llm.AnthropicReasoner()
    world = await make_world()
    tools = ToolRegistry()

    @tools.register(name="clock.now", description="wall clock", returns="str")
    async def _now() -> str:
        return "now"

    await r.answer(Intent(text="hi"), world, tools)
    context_text = client.calls[0]["system"][1]["text"]
    assert "AVAILABLE TOOLS" in context_text
    assert "clock.now" in context_text


async def test_user_message_carries_intent_text(monkeypatch):
    client = _install_fake(monkeypatch)
    r = llm.AnthropicReasoner()
    world = await make_world()
    await r.answer(Intent(text="specific prompt xyz"), world, ToolRegistry())
    messages = client.calls[0]["messages"]
    assert messages == [{"role": "user", "content": "specific prompt xyz"}]


async def test_happy_path_returns_high_confidence(monkeypatch):
    _install_fake(monkeypatch, canned_text="the answer", canned_stop="end_turn")
    r = llm.AnthropicReasoner()
    world = await make_world()
    resp = await r.answer(Intent(text="hi"), world, ToolRegistry())
    assert resp.text == "the answer"
    assert resp.confidence == pytest.approx(0.85)
    assert resp.tier == "deep"


async def test_non_end_turn_lowers_confidence(monkeypatch):
    _install_fake(monkeypatch, canned_text="partial", canned_stop="max_tokens")
    r = llm.AnthropicReasoner()
    world = await make_world()
    resp = await r.answer(Intent(text="hi"), world, ToolRegistry())
    assert resp.confidence == pytest.approx(0.5)


async def test_api_error_returns_zero_confidence(monkeypatch):
    _install_fake(monkeypatch, raise_exc=RuntimeError("boom"))
    r = llm.AnthropicReasoner()
    world = await make_world()
    resp = await r.answer(Intent(text="hi"), world, ToolRegistry())
    assert resp.confidence == 0.0
    assert "error" in resp.text.lower()


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="real-API integration test; needs ANTHROPIC_API_KEY",
)
async def test_real_api_smoke():  # pragma: no cover
    # Only runs when a real key is present; gated per unit spec.
    import importlib

    importlib.reload(llm)
    r = llm.AnthropicReasoner(max_tokens=64)
    world = await make_world()
    resp = await r.answer(Intent(text="say 'ok'"), world, ToolRegistry())
    assert resp.text
