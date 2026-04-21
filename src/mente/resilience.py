"""Thin resilience primitives: retry, timeout, circuit breaker.

Pure-stdlib wrappers applied surgically at a handful of call sites — the LLM
API call and the synthesis subprocess await. Keeps the "thin seed" ethos:
no third-party deps, no structural refactors, happy path bitwise identical.
"""
from __future__ import annotations

import asyncio
import functools
import random
import time
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")

_DEFAULT_RETRY_ON: tuple[type[BaseException], ...] = (ConnectionError, TimeoutError)


def retry_async(
    attempts: int = 3,
    backoff: float = 0.2,
    jitter: float = 0.1,
    retry_on: tuple[type[BaseException], ...] = _DEFAULT_RETRY_ON,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator: retry an async callable with exponential backoff + jitter.

    Retries only on `retry_on` types; other exceptions propagate immediately.
    Re-raises the last caught exception once attempts are exhausted.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc: BaseException | None = None
            for i in range(attempts):
                try:
                    return await fn(*args, **kwargs)
                except retry_on as e:  # type: ignore[misc]
                    last_exc = e
                    if i == attempts - 1:
                        break
                    delay = backoff * (2 ** i) + random.uniform(0.0, jitter)
                    await asyncio.sleep(delay)
            assert last_exc is not None  # loop guarantees at least one exception
            raise last_exc

        return wrapper

    return decorator


def timeout(
    seconds: float,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator: enforce an asyncio timeout on an async callable.

    Raises asyncio.TimeoutError with a clear message on expiry.
    """

    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return await asyncio.wait_for(fn(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError as e:
                raise asyncio.TimeoutError(
                    f"{getattr(fn, '__name__', 'call')} exceeded {seconds}s timeout"
                ) from e

        return wrapper

    return decorator


class CircuitOpenError(RuntimeError):
    """Raised when a CircuitBreaker is open and refuses the call."""


class CircuitBreaker:
    """Tiny closed/open/half-open breaker.

    - closed: calls pass through; consecutive failures increment a counter.
    - open: calls immediately raise CircuitOpenError until recovery_s elapses.
    - half-open: a single probe is allowed; success closes, failure re-opens.
    """

    def __init__(self, failure_threshold: int = 5, recovery_s: float = 60.0) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self.failure_threshold = failure_threshold
        self.recovery_s = recovery_s
        self._failures = 0
        self._opened_at: float | None = None
        self._state: str = "closed"

    @property
    def state(self) -> str:
        self._maybe_transition_to_half_open()
        return self._state

    def _maybe_transition_to_half_open(self) -> None:
        if (
            self._state == "open"
            and self._opened_at is not None
            and time.monotonic() - self._opened_at >= self.recovery_s
        ):
            self._state = "half_open"

    async def call(self, func: Callable[[], Awaitable[T]]) -> T:
        self._maybe_transition_to_half_open()
        if self._state == "open":
            raise CircuitOpenError("circuit breaker is open")
        try:
            result = await func()
        except Exception:
            self._on_failure()
            raise
        self._on_success()
        return result

    def _on_success(self) -> None:
        self._failures = 0
        self._opened_at = None
        self._state = "closed"

    def _on_failure(self) -> None:
        if self._state == "half_open":
            self._state = "open"
            self._opened_at = time.monotonic()
            return
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._state = "open"
            self._opened_at = time.monotonic()
