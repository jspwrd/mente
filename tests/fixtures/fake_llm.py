"""Reusable fake Anthropic async client for testing mente.llm.AnthropicReasoner.

Provides a drop-in replacement for `anthropic.AsyncAnthropic` that records
every `messages.create(...)` call and returns canned content blocks. Tests can
use this to assert system-prompt structure, model selection, thinking
parameters, cache breakpoints, and error paths without hitting the real API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeMessage:
    content: list[FakeTextBlock]
    stop_reason: str = "end_turn"


@dataclass
class FakeMessagesAPI:
    """Captures each call and returns the configured response."""
    parent: "FakeAsyncAnthropic"

    async def create(self, **kwargs: Any) -> FakeMessage:
        self.parent.calls.append(kwargs)
        if self.parent.raise_exc is not None:
            raise self.parent.raise_exc
        return self.parent.canned_response


@dataclass
class FakeAsyncAnthropic:
    """Minimal fake implementing only what AnthropicReasoner uses."""
    canned_text: str = "canned answer"
    canned_stop: str = "end_turn"
    raise_exc: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.messages = FakeMessagesAPI(parent=self)

    @property
    def canned_response(self) -> FakeMessage:
        return FakeMessage(
            content=[FakeTextBlock(text=self.canned_text)],
            stop_reason=self.canned_stop,
        )


def make_fake_anthropic_module(
    canned_text: str = "canned answer",
    canned_stop: str = "end_turn",
    raise_exc: Exception | None = None,
) -> tuple[Any, FakeAsyncAnthropic]:
    """Build a fake `anthropic` module stand-in + the FakeAsyncAnthropic instance
    its `AsyncAnthropic()` constructor will return.

    Use this to monkeypatch `mente.llm.anthropic` so AnthropicReasoner.__post_init__
    picks up the fake without the real SDK being installed.
    """
    fake_client = FakeAsyncAnthropic(
        canned_text=canned_text,
        canned_stop=canned_stop,
        raise_exc=raise_exc,
    )

    class _FakeAnthropicModule:
        AsyncAnthropic = staticmethod(lambda *a, **kw: fake_client)

    return _FakeAnthropicModule(), fake_client
