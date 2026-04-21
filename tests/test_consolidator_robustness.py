"""Unit 12: robustness tests for Consolidator.

Covers the three edge cases the unit promises to harden:
  - empty SlowMemory must not raise ZeroDivisionError
  - pure-note history must yield sane defaults (accept_rate == 1.0)
  - a transient failure inside digest() must NOT kill the run() loop
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

# Make src/ importable when pytest is invoked from the project root.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aria.consolidator import Consolidator  # noqa: E402
from aria.memory import SlowMemory  # noqa: E402
from aria.state import LatentState  # noqa: E402


def _make_consolidator(tmp_path: Path) -> Consolidator:
    slow = SlowMemory(db_path=tmp_path / "slow.sqlite")
    latent = LatentState(values={}, path=None)  # no checkpoint write
    return Consolidator(slow_mem=slow, latent=latent, interval_s=0.01)


def test_digest_on_empty_memory_returns_defaults(tmp_path: Path) -> None:
    c = _make_consolidator(tmp_path)
    try:
        d = c.digest()
        # Schema keys required by self_model.answer() must remain present.
        assert "total_responses" in d
        assert "accept_rate" in d
        assert "by_reasoner" in d
        # Zero/defaults (no division-by-zero).
        assert d["total_responses"] == 0
        assert d["note_count"] == 0
        assert d["by_reasoner"] == {}
        assert d["accept_rate"] == 1.0  # vacuous truth: no rejections
        assert d["avg_verdict_score"] == 0.0
        assert d["recent_notes"] == []
        # New optional summary key present.
        assert "summary" in d
        assert d["summary"]["total"] == 0
    finally:
        c.slow_mem.close()


def test_consolidate_on_empty_memory_succeeds(tmp_path: Path) -> None:
    c = _make_consolidator(tmp_path)
    try:
        d = c.consolidate()
        assert isinstance(d, dict)
        assert d.get("total_responses") == 0
        assert "error" not in d
    finally:
        c.slow_mem.close()


def test_digest_with_only_notes_accepts_defaults(tmp_path: Path) -> None:
    c = _make_consolidator(tmp_path)
    try:
        for i in range(3):
            c.slow_mem.record("note", "curiosity", {"fact": f"fact-{i}"})
        d = c.digest()
        assert d["total_responses"] == 0
        assert d["note_count"] == 3
        assert d["accept_rate"] == 1.0  # no verdicts → default
        assert d["avg_verdict_score"] == 0.0
        assert d["by_reasoner"] == {}
        assert set(d["recent_notes"]) == {"fact-0", "fact-1", "fact-2"}
    finally:
        c.slow_mem.close()


def test_recent_notes_capped_at_ten(tmp_path: Path) -> None:
    c = _make_consolidator(tmp_path)
    try:
        for i in range(25):
            c.slow_mem.record("note", "curiosity", {"fact": f"n{i}"})
        d = c.digest()
        assert len(d["recent_notes"]) == 10
    finally:
        c.slow_mem.close()


def test_consolidate_swallows_exception_returns_error(tmp_path: Path) -> None:
    c = _make_consolidator(tmp_path)
    try:
        # Force digest() to blow up.
        def boom() -> dict:
            raise RuntimeError("synthetic")

        c.digest = boom  # type: ignore[assignment]
        out = c.consolidate()
        # No prior digest exists → error marker returned.
        assert out.get("error") == "synthetic"
    finally:
        c.slow_mem.close()


def test_consolidate_falls_back_to_last_known_digest(tmp_path: Path) -> None:
    c = _make_consolidator(tmp_path)
    try:
        good = c.consolidate()
        assert good.get("total_responses") == 0

        def boom() -> dict:
            raise RuntimeError("later failure")

        c.digest = boom  # type: ignore[assignment]
        out = c.consolidate()
        # Should return the last successful digest, not an error.
        assert "error" not in out
        assert out.get("total_responses") == 0
    finally:
        c.slow_mem.close()


def test_run_loop_survives_digest_exception(tmp_path: Path) -> None:
    c = _make_consolidator(tmp_path)
    try:
        calls = {"n": 0}
        real_digest = c.digest

        def flaky() -> dict:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("one-shot failure")
            return real_digest()

        c.digest = flaky  # type: ignore[assignment]

        async def drive() -> None:
            stop = asyncio.Event()
            task = asyncio.create_task(c.run(stop))
            # Let the loop tick twice (interval_s = 0.01 in fixture).
            await asyncio.sleep(0.08)
            stop.set()
            await asyncio.wait_for(task, timeout=1.0)

        asyncio.run(drive())
        # Must have made it past the raising call at least once.
        assert calls["n"] >= 2
    finally:
        c.slow_mem.close()
