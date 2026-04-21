"""Tests for mente.resilience."""
from __future__ import annotations

import asyncio

import pytest

from mente.resilience import CircuitBreaker, CircuitOpenError, retry_async, timeout


# --- retry_async ----------------------------------------------------------


def test_retry_async_succeeds_first_try() -> None:
    calls = {"n": 0}

    @retry_async(attempts=3, backoff=0.001, jitter=0.0)
    async def ok() -> str:
        calls["n"] += 1
        return "ok"

    assert asyncio.run(ok()) == "ok"
    assert calls["n"] == 1


def test_retry_async_recovers_after_failures() -> None:
    calls = {"n": 0}

    @retry_async(attempts=3, backoff=0.001, jitter=0.0,
                 retry_on=(ConnectionError,))
    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("boom")
        return "done"

    assert asyncio.run(flaky()) == "done"
    assert calls["n"] == 3


def test_retry_async_exhausts_and_raises() -> None:
    calls = {"n": 0}

    @retry_async(attempts=3, backoff=0.001, jitter=0.0,
                 retry_on=(ConnectionError,))
    async def always_bad() -> str:
        calls["n"] += 1
        raise ConnectionError(f"attempt {calls['n']}")

    with pytest.raises(ConnectionError, match="attempt 3"):
        asyncio.run(always_bad())
    assert calls["n"] == 3


def test_retry_async_filter_does_not_retry_other_types() -> None:
    calls = {"n": 0}

    @retry_async(attempts=3, backoff=0.001, jitter=0.0,
                 retry_on=(ConnectionError,))
    async def bad() -> None:
        calls["n"] += 1
        raise ValueError("nope")

    with pytest.raises(ValueError):
        asyncio.run(bad())
    assert calls["n"] == 1


# --- timeout --------------------------------------------------------------


def test_timeout_passes_through_fast_call() -> None:
    @timeout(0.5)
    async def fast() -> int:
        await asyncio.sleep(0.0)
        return 42

    assert asyncio.run(fast()) == 42


def test_timeout_raises_on_slow_call() -> None:
    @timeout(0.05)
    async def slow() -> None:
        await asyncio.sleep(1.0)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(slow())


# --- CircuitBreaker -------------------------------------------------------


def test_circuit_breaker_opens_after_threshold() -> None:
    cb = CircuitBreaker(failure_threshold=3, recovery_s=60.0)

    async def boom() -> None:
        raise ConnectionError("nope")

    async def run() -> None:
        for _ in range(3):
            with pytest.raises(ConnectionError):
                await cb.call(boom)
        # Now open: calls should short-circuit.
        with pytest.raises(CircuitOpenError):
            await cb.call(boom)

    asyncio.run(run())
    assert cb.state == "open"


def test_circuit_breaker_half_open_recovers_on_success() -> None:
    cb = CircuitBreaker(failure_threshold=2, recovery_s=0.05)

    async def boom() -> None:
        raise ConnectionError("nope")

    async def ok() -> str:
        return "ok"

    async def run() -> None:
        for _ in range(2):
            with pytest.raises(ConnectionError):
                await cb.call(boom)
        assert cb.state == "open"
        # Wait past recovery window.
        await asyncio.sleep(0.08)
        # State read transitions to half_open.
        assert cb.state == "half_open"
        result = await cb.call(ok)
        assert result == "ok"
        assert cb.state == "closed"

    asyncio.run(run())


def test_circuit_breaker_half_open_reopens_on_failure() -> None:
    cb = CircuitBreaker(failure_threshold=1, recovery_s=0.02)

    async def boom() -> None:
        raise ConnectionError("nope")

    async def run() -> None:
        with pytest.raises(ConnectionError):
            await cb.call(boom)
        assert cb.state == "open"
        await asyncio.sleep(0.04)
        assert cb.state == "half_open"
        with pytest.raises(ConnectionError):
            await cb.call(boom)
        assert cb.state == "open"

    asyncio.run(run())
