"""LLM synthesizer — exercised with a faked ``anthropic.AsyncAnthropic`` so
no network or live API key is required. A single live-API test is gated
behind ``ANTHROPIC_API_KEY``.
"""
from __future__ import annotations

import asyncio
import importlib
import os
import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fixtures — install a fake `anthropic` module (if absent) and reload the
# synthesizer so it picks up the stub.
# ---------------------------------------------------------------------------


def _install_fake_anthropic(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Ensure a module named ``anthropic`` exists with an AsyncAnthropic
    attribute; return it. If the real SDK is installed we leave it alone."""
    if "anthropic" not in sys.modules:
        fake = types.ModuleType("anthropic")

        class _FakeAsyncAnthropic:  # minimal surface LLMSynthesizer calls
            def __init__(self, *a: Any, **kw: Any) -> None:
                self.messages = MagicMock()

        fake.AsyncAnthropic = _FakeAsyncAnthropic  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "anthropic", fake)
    return sys.modules["anthropic"]


def _reload_llm_module() -> Any:
    """Reload mente.synthesizers.llm so its module-level anthropic import
    rebinds to the current ``sys.modules['anthropic']``."""
    if "mente.synthesizers.llm" in sys.modules:
        return importlib.reload(sys.modules["mente.synthesizers.llm"])
    return importlib.import_module("mente.synthesizers.llm")


def _fake_message(text: str) -> Any:
    block = types.SimpleNamespace(type="text", text=text)
    return types.SimpleNamespace(
        content=[block], stop_reason="end_turn",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_missing_sdk_raises_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``anthropic`` can't be imported, constructing LLMSynthesizer
    raises RuntimeError with a clean install hint (no ImportError leak)."""
    import builtins as _builtins

    monkeypatch.delitem(sys.modules, "anthropic", raising=False)
    real_import = _builtins.__import__

    def guarded_import(name: str, *a: Any, **kw: Any) -> Any:
        if name == "anthropic" or name.startswith("anthropic."):
            raise ImportError("no anthropic for this test")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(_builtins, "__import__", guarded_import)
    llm_mod = _reload_llm_module()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    with pytest.raises(RuntimeError, match="anthropic SDK not installed"):
        llm_mod.LLMSynthesizer()


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_anthropic(monkeypatch)
    llm_mod = _reload_llm_module()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY not set"):
        llm_mod.LLMSynthesizer()


def test_synthesize_parses_json_response(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_anthropic(monkeypatch)
    llm_mod = _reload_llm_module()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    synth = llm_mod.LLMSynthesizer()
    canned = _fake_message(
        '{"source": "def fib(n):\\n    a, b = 0, 1\\n    for _ in range(n):\\n'
        '        a, b = b, a + b\\n    return a\\n", '
        '"entrypoint": "fib", "args": {"n": 10}}'
    )
    synth._client.messages.create = AsyncMock(return_value=canned)

    out = asyncio.run(synth.asynthesize("compute the 10th fibonacci number"))
    assert out is not None
    source, entry, args = out
    assert entry == "fib"
    assert args == {"n": 10}
    assert "def fib(n):" in source


def test_synthesize_strips_fenced_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_anthropic(monkeypatch)
    llm_mod = _reload_llm_module()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    synth = llm_mod.LLMSynthesizer()
    canned = _fake_message(
        "```json\n"
        '{"source": "def add(a, b):\\n    return a + b\\n", '
        '"entrypoint": "add", "args": {"a": 2, "b": 3}}\n'
        "```"
    )
    synth._client.messages.create = AsyncMock(return_value=canned)

    out = asyncio.run(synth.asynthesize("add 2 and 3"))
    assert out is not None
    source, entry, args = out
    assert entry == "add"
    assert args == {"a": 2, "b": 3}
    assert "return a + b" in source


def test_synthesize_extracts_embedded_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_anthropic(monkeypatch)
    llm_mod = _reload_llm_module()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    synth = llm_mod.LLMSynthesizer()
    canned = _fake_message(
        'Here is the function:\n\n'
        '{"source": "def sq(x):\\n    return x * x\\n", '
        '"entrypoint": "sq", "args": {"x": 7}}\n\nHope that helps.'
    )
    synth._client.messages.create = AsyncMock(return_value=canned)

    out = asyncio.run(synth.asynthesize("square 7"))
    assert out is not None
    assert out[1] == "sq"
    assert out[2] == {"x": 7}


def test_synthesize_refusal_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_anthropic(monkeypatch)
    llm_mod = _reload_llm_module()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    synth = llm_mod.LLMSynthesizer()
    synth._client.messages.create = AsyncMock(
        return_value=_fake_message('{"refuse": true}')
    )
    assert asyncio.run(synth.asynthesize("hack the mainframe")) is None


def test_synthesize_bad_json_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_anthropic(monkeypatch)
    llm_mod = _reload_llm_module()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    synth = llm_mod.LLMSynthesizer()
    synth._client.messages.create = AsyncMock(
        return_value=_fake_message("definitely not json")
    )
    assert asyncio.run(synth.asynthesize("compute something")) is None


def test_synthesize_missing_keys_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_anthropic(monkeypatch)
    llm_mod = _reload_llm_module()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    synth = llm_mod.LLMSynthesizer()
    synth._client.messages.create = AsyncMock(
        return_value=_fake_message('{"entrypoint": "f"}')  # no source
    )
    assert asyncio.run(synth.asynthesize("x")) is None


def test_synthesize_api_exception_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_anthropic(monkeypatch)
    llm_mod = _reload_llm_module()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    synth = llm_mod.LLMSynthesizer()
    synth._client.messages.create = AsyncMock(side_effect=RuntimeError("boom"))
    assert asyncio.run(synth.asynthesize("x")) is None


def test_sync_synthesize_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_anthropic(monkeypatch)
    llm_mod = _reload_llm_module()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    synth = llm_mod.LLMSynthesizer()
    canned = _fake_message(
        '{"source": "def f():\\n    return 1\\n", '
        '"entrypoint": "f", "args": {}}'
    )
    synth._client.messages.create = AsyncMock(return_value=canned)

    out = synth.synthesize("trivial")  # no loop running
    assert out is not None
    assert out[1] == "f"


def test_sync_synthesize_from_running_loop_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_anthropic(monkeypatch)
    llm_mod = _reload_llm_module()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    synth = llm_mod.LLMSynthesizer()

    async def _from_loop() -> None:
        with pytest.raises(RuntimeError, match="running event loop"):
            synth.synthesize("x")

    asyncio.run(_from_loop())


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="live API key required",
)
def test_live_synthesis() -> None:  # pragma: no cover — opt-in
    from mente.synthesizers.llm import LLMSynthesizer

    synth = LLMSynthesizer()
    out = asyncio.run(synth.asynthesize("compute the 10th fibonacci number"))
    assert out is not None
    assert out[1]  # some entrypoint name
