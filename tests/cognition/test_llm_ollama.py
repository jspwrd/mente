"""Tests for :mod:`mente.llm_ollama` — constructor gating, prompt shape,
and all of the failure modes. Never calls a real Ollama server (except the
explicitly gated live-integration smoke test).
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fixtures.cognition_helpers import make_world

from mente import llm_ollama
from mente.tools import ToolRegistry
from mente.types import Belief, Intent

# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Shape-compatible stand-in for an ``httpx.Response``."""

    def __init__(
        self,
        status_code: int = 200,
        json_data: Any = None,
        raise_on_json: Exception | None = None,
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data
        self._raise_on_json = raise_on_json

    def json(self) -> Any:
        if self._raise_on_json is not None:
            raise self._raise_on_json
        return self._json_data


def _fake_async_client_factory(
    post_result: Any | None = None,
    post_exc: Exception | None = None,
) -> tuple[MagicMock, AsyncMock]:
    """Build an ``AsyncClient(...)`` factory whose ``post`` is mockable.

    Returns:
        ``(factory, post_mock)`` — pass ``factory`` into the module's
        ``httpx.AsyncClient`` slot; assert on ``post_mock`` for call inspection.
    """
    post_mock = AsyncMock()
    if post_exc is not None:
        post_mock.side_effect = post_exc
    else:
        post_mock.return_value = post_result

    @asynccontextmanager
    async def _ctx(*args: Any, **kwargs: Any):
        client = MagicMock()
        client.post = post_mock
        yield client

    factory = MagicMock(side_effect=lambda *a, **kw: _ctx(*a, **kw))
    return factory, post_mock


def _install_fake_httpx(
    monkeypatch: pytest.MonkeyPatch,
    post_result: Any | None = None,
    post_exc: Exception | None = None,
) -> AsyncMock:
    """Wire a fake ``httpx`` module onto the reasoner's ``_client`` slot.

    The module's lazy import is bypassed by replacing ``_load_httpx`` — we do
    not need the real ``httpx`` installed for unit tests.
    """
    factory, post_mock = _fake_async_client_factory(
        post_result=post_result, post_exc=post_exc
    )
    fake_httpx = MagicMock()
    fake_httpx.AsyncClient = factory
    monkeypatch.setattr(llm_ollama, "_load_httpx", lambda: fake_httpx)
    return post_mock


# ---------------------------------------------------------------------------
# constructor + ollama_available
# ---------------------------------------------------------------------------


def test_constructor_raises_when_httpx_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> Any:
        raise ImportError(llm_ollama._IMPORT_ERROR_MSG)

    monkeypatch.setattr(llm_ollama, "_load_httpx", _raise)
    with pytest.raises(ImportError, match="mente\\[llm-ollama\\]"):
        llm_ollama.OllamaReasoner()


def test_ollama_available_true_on_200(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_httpx = MagicMock()
    fake_httpx.get = MagicMock(return_value=MagicMock(status_code=200))
    monkeypatch.setattr(llm_ollama, "_load_httpx", lambda: fake_httpx)
    assert llm_ollama.ollama_available("http://127.0.0.1:11434") is True
    fake_httpx.get.assert_called_once()
    # Ensure the probe uses a 1-second timeout (not the reasoner's 30s).
    _args, kwargs = fake_httpx.get.call_args
    assert kwargs.get("timeout") == 1.0


def test_ollama_available_false_on_non_200(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_httpx = MagicMock()
    fake_httpx.get = MagicMock(return_value=MagicMock(status_code=500))
    monkeypatch.setattr(llm_ollama, "_load_httpx", lambda: fake_httpx)
    assert llm_ollama.ollama_available() is False


def test_ollama_available_false_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_httpx = MagicMock()
    fake_httpx.get = MagicMock(side_effect=ConnectionError("refused"))
    monkeypatch.setattr(llm_ollama, "_load_httpx", lambda: fake_httpx)
    assert llm_ollama.ollama_available() is False


def test_ollama_available_false_when_httpx_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise() -> Any:
        raise ImportError(llm_ollama._IMPORT_ERROR_MSG)

    monkeypatch.setattr(llm_ollama, "_load_httpx", _raise)
    assert llm_ollama.ollama_available() is False


# ---------------------------------------------------------------------------
# request shape
# ---------------------------------------------------------------------------


async def test_happy_path_returns_high_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _FakeResponse(
        status_code=200,
        json_data={"message": {"content": "the answer"}, "done": True},
    )
    _install_fake_httpx(monkeypatch, post_result=response)
    r = llm_ollama.OllamaReasoner()
    world = await make_world()
    resp = await r.answer(Intent(text="hi"), world, ToolRegistry())
    assert resp.text == "the answer"
    assert resp.confidence == pytest.approx(0.75)
    assert resp.tier == "deep"
    assert resp.reasoner == "deep.ollama"


async def test_request_hits_chat_endpoint_with_expected_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _FakeResponse(
        status_code=200,
        json_data={"message": {"content": "ok"}, "done": True},
    )
    post = _install_fake_httpx(monkeypatch, post_result=response)
    r = llm_ollama.OllamaReasoner(model="llama3.2", url="http://127.0.0.1:11434")
    world = await make_world([Belief(entity="user", attribute="name", value="Ada")])
    await r.answer(Intent(text="who am I"), world, ToolRegistry())
    assert post.call_count == 1
    args, kwargs = post.call_args
    # First positional arg is the endpoint.
    assert args[0] == "http://127.0.0.1:11434/api/chat"
    body = kwargs["json"]
    assert body["model"] == "llama3.2"
    assert body["stream"] is False
    messages = body["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "MENTE" in messages[0]["content"]
    assert "WORLD MODEL" in messages[0]["content"]
    assert "user.name" in messages[0]["content"]
    assert "Ada" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "who am I"}


async def test_system_prompt_lists_available_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _FakeResponse(
        status_code=200,
        json_data={"message": {"content": "ok"}, "done": True},
    )
    post = _install_fake_httpx(monkeypatch, post_result=response)
    r = llm_ollama.OllamaReasoner()
    world = await make_world()
    tools = ToolRegistry()

    @tools.register(name="clock.now", description="wall clock", returns="str")
    async def _now() -> str:
        return "now"

    await r.answer(Intent(text="hi"), world, tools)
    system_prompt = post.call_args.kwargs["json"]["messages"][0]["content"]
    assert "AVAILABLE TOOLS" in system_prompt
    assert "clock.now" in system_prompt


# ---------------------------------------------------------------------------
# failure modes
# ---------------------------------------------------------------------------


async def test_transport_timeout_returns_zero_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_httpx(monkeypatch, post_exc=TimeoutError("timed out"))
    r = llm_ollama.OllamaReasoner()
    world = await make_world()
    resp = await r.answer(Intent(text="hi"), world, ToolRegistry())
    assert resp.confidence == 0.0
    assert "error" in resp.text.lower()
    assert "TimeoutError" in resp.text
    assert resp.reasoner == "deep.ollama"
    assert resp.tier == "deep"


async def test_non_200_returns_zero_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _FakeResponse(status_code=500, json_data={"error": "nope"})
    _install_fake_httpx(monkeypatch, post_result=response)
    r = llm_ollama.OllamaReasoner()
    world = await make_world()
    resp = await r.answer(Intent(text="hi"), world, ToolRegistry())
    assert resp.confidence == 0.0
    assert "HTTP 500" in resp.text


async def test_malformed_json_returns_zero_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _FakeResponse(
        status_code=200,
        raise_on_json=ValueError("not json"),
    )
    _install_fake_httpx(monkeypatch, post_result=response)
    r = llm_ollama.OllamaReasoner()
    world = await make_world()
    resp = await r.answer(Intent(text="hi"), world, ToolRegistry())
    assert resp.confidence == 0.0
    assert "malformed JSON" in resp.text


async def test_missing_message_content_returns_zero_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _FakeResponse(
        status_code=200,
        json_data={"done": True},  # no "message" key
    )
    _install_fake_httpx(monkeypatch, post_result=response)
    r = llm_ollama.OllamaReasoner()
    world = await make_world()
    resp = await r.answer(Intent(text="hi"), world, ToolRegistry())
    assert resp.confidence == 0.0
    assert "missing message.content" in resp.text


async def test_empty_message_content_returns_zero_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _FakeResponse(
        status_code=200,
        json_data={"message": {"content": "   "}, "done": True},
    )
    _install_fake_httpx(monkeypatch, post_result=response)
    r = llm_ollama.OllamaReasoner()
    world = await make_world()
    resp = await r.answer(Intent(text="hi"), world, ToolRegistry())
    assert resp.confidence == 0.0
    assert "missing message.content" in resp.text


# ---------------------------------------------------------------------------
# logging guards
# ---------------------------------------------------------------------------


async def test_logs_never_carry_intent_text(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _install_fake_httpx(monkeypatch, post_exc=ConnectionError("refused"))
    r = llm_ollama.OllamaReasoner()
    world = await make_world()
    # Make sure our logger propagates for caplog — the mente logger is
    # configured with propagate=False elsewhere, so attach a direct handler.
    import logging

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture(level=logging.DEBUG)
    logger = logging.getLogger("mente.llm_ollama")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        secret = "super-secret-intent-abcdef"
        await r.answer(Intent(text=secret), world, ToolRegistry())
        for rec in records:
            assert secret not in rec.getMessage()
    finally:
        logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# live integration (gated)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("OLLAMA_URL"),
    reason="live Ollama integration; set OLLAMA_URL to run",
)
async def test_live_ollama_smoke() -> None:  # pragma: no cover
    # Uses whatever model is resident — the user owns the tradeoff.
    url = os.environ["OLLAMA_URL"]
    model = os.environ.get("OLLAMA_MODEL", "llama3.2")
    # Sanity: make sure httpx actually imports here.
    with patch.object(llm_ollama, "_load_httpx", wraps=llm_ollama._load_httpx):
        r = llm_ollama.OllamaReasoner(url=url, model=model, max_tokens=64)
    world = await make_world()
    resp = await r.answer(Intent(text="say 'ok' and nothing else"), world, ToolRegistry())
    assert resp.text
