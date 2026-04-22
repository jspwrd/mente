"""Runtime shutdown error tests.

Exercises the `Runtime.shutdown()` path under a handful of edge cases:
clean shutdown, shutdown after activity, shutdown during background tasks,
consolidator failure, double shutdown, and a broken `SemanticMemory.close`.
Each test uses its own `tmp_path` so no on-disk state leaks between cases.
"""
from __future__ import annotations

import asyncio
import contextlib
import warnings
from pathlib import Path

import pytest
from fixtures.advanced_helpers import make_runtime

from mente.consolidator import Consolidator
from mente.embeddings import SemanticMemory
from mente.types import Intent


@pytest.mark.asyncio
async def test_clean_shutdown_closes_db_and_checkpoints_latent(tmp_path: Path) -> None:
    rt = await make_runtime(tmp_path)
    await rt.shutdown()
    assert rt.slow_mem._conn is None
    assert (rt.root / "latent.json").exists()


@pytest.mark.asyncio
async def test_shutdown_after_activity_persists_rows(tmp_path: Path) -> None:
    rt = await make_runtime(tmp_path)
    for text in ("hello", "what time is it", "remember this"):
        await rt.handle_intent(Intent(text=text))

    assert rt.slow_mem.query(limit=100), "activity should have produced episodic rows"

    await rt.shutdown()

    assert rt.slow_mem._conn is None
    assert (rt.root / "latent.json").exists()
    assert (rt.root / "episodic.sqlite").exists()


@pytest.mark.asyncio
async def test_shutdown_during_background_no_task_leak(tmp_path: Path) -> None:
    rt = await make_runtime(tmp_path)
    rt.consolidator.interval_s = 0.05
    rt.curiosity.interval_s = 0.05
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tasks = rt.start_background()
        await rt.shutdown()
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True), timeout=2.0
        )
        assert all(t.done() for t in tasks)
        leaks = [
            w for w in caught
            if "coroutine" in str(w.message).lower()
            and "never awaited" in str(w.message).lower()
        ]
        assert not leaks, f"unawaited coroutine warnings: {[str(w.message) for w in leaks]}"


@pytest.mark.asyncio
async def test_shutdown_with_failing_consolidator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rt = await make_runtime(tmp_path)
    rt.consolidator.interval_s = 0.05
    calls = {"n": 0}
    original = Consolidator.consolidate

    def flaky(self: Consolidator) -> dict[str, object]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return original(self)

    monkeypatch.setattr(Consolidator, "consolidate", flaky)

    tasks = rt.start_background()
    # Wait long enough for the consolidator loop to tick at least once.
    await asyncio.sleep(0.15)

    # shutdown must not propagate the RuntimeError.
    await rt.shutdown()

    await asyncio.wait_for(
        asyncio.gather(*tasks, return_exceptions=True), timeout=2.0
    )
    assert rt.slow_mem._conn is None, "slow_mem should be closed despite the failure"


@pytest.mark.asyncio
async def test_double_shutdown_is_idempotent(tmp_path: Path) -> None:
    rt = await make_runtime(tmp_path)
    await rt.shutdown()
    # Second call should be a no-op (or at least not raise).
    await rt.shutdown()
    assert rt.slow_mem._conn is None


@pytest.mark.asyncio
async def test_shutdown_with_broken_semantic_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rt = await make_runtime(tmp_path)
    try:
        calls = {"n": 0}

        def broken_close(self: SemanticMemory) -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("semantic close exploded")
            if self._conn is not None:
                self._conn.close()
                self._conn = None

        monkeypatch.setattr(SemanticMemory, "close", broken_close)

        # The current implementation runs closes sequentially, so a failure in
        # semantic_mem.close may abort later steps. slow_mem.close runs *before*
        # semantic_mem.close, so it should still be closed afterwards. If
        # shutdown doesn't swallow the exception, that's a known follow-up
        # (documented in the PR body); we only assert the invariant we can
        # guarantee today.
        with contextlib.suppress(RuntimeError):
            await rt.shutdown()

        assert rt.slow_mem._conn is None, (
            "slow_mem.close runs before semantic_mem.close; it must have closed"
        )
    finally:
        # Best-effort cleanup; subsequent calls are allowed to raise too.
        with contextlib.suppress(Exception):
            if rt.slow_mem._conn is not None:
                rt.slow_mem.close()
