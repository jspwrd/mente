"""Tests for the consolidation loop.

digest() aggregates response rows into per-reasoner counts, accept rate,
average verdict score, and recent notes. consolidate() records a digest row
and stashes it into LatentState.last_digest.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mente.consolidator import Consolidator
from mente.memory import SlowMemory
from mente.state import LatentState


def _setup(tmp_path: Path) -> tuple[SlowMemory, LatentState, Consolidator]:
    slow = SlowMemory(db_path=tmp_path / "episodic.sqlite")
    latent = LatentState(path=tmp_path / "latent.json")
    return slow, latent, Consolidator(slow_mem=slow, latent=latent)


def _seed_responses(slow: SlowMemory) -> None:
    slow.record("response", "fast.heuristic", {"text": "hi", "verdict": {"accept": True, "score": 0.9}})
    slow.record("response", "fast.heuristic", {"text": "ok", "verdict": {"accept": False, "score": 0.3}})
    slow.record("response", "deep.sim", {"text": "hmm", "verdict": {"accept": True, "score": 0.6}})
    slow.record("note", "user", {"fact": "redis uses AOF"})
    slow.record("note", "user", {"fact": "postgres uses WAL"})


def test_digest_aggregates_by_reasoner_accept_rate_and_notes(tmp_path: Path) -> None:
    slow, latent, cons = _setup(tmp_path)
    try:
        _seed_responses(slow)
        d = cons.digest()

        assert d["total_responses"] == 3
        assert d["by_reasoner"] == {"fast.heuristic": 2, "deep.sim": 1}
        # 2 of 3 accepted → 0.667
        assert 0.66 <= d["accept_rate"] <= 0.68
        # Average verdict score of (0.9 + 0.3 + 0.6) / 3 = 0.6
        assert d["avg_verdict_score"] == pytest.approx(0.6, abs=0.01)
        assert d["note_count"] == 2
        assert "redis uses AOF" in d["recent_notes"]
        assert "postgres uses WAL" in d["recent_notes"]
    finally:
        slow.close()


def test_digest_on_empty_store_has_safe_defaults(tmp_path: Path) -> None:
    slow, latent, cons = _setup(tmp_path)
    try:
        d = cons.digest()
        assert d["total_responses"] == 0
        assert d["by_reasoner"] == {}
        # With no verdicts, accept_rate defaults to 1.0.
        assert d["accept_rate"] == 1.0
        assert d["avg_verdict_score"] == 0.0
        assert d["note_count"] == 0
        assert d["recent_notes"] == []
    finally:
        slow.close()


def test_consolidate_records_digest_and_updates_latent(tmp_path: Path) -> None:
    slow, latent, cons = _setup(tmp_path)
    try:
        _seed_responses(slow)
        d = cons.consolidate()

        # Latent.last_digest is populated.
        assert latent.get("last_digest") == d

        # A "digest" kind row was written to slow_mem.
        rows = slow.query(kind="digest", limit=5)
        assert rows
        assert rows[0]["actor"] == "consolidator"
        assert rows[0]["payload"]["total_responses"] == d["total_responses"]

        # Checkpoint was written to disk.
        assert latent.path is not None
        assert latent.path.exists()
    finally:
        slow.close()
