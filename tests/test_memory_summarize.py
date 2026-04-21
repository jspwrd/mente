"""Unit 12: SlowMemory.summarize() — aggregated stats helper."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mente.memory import SlowMemory  # noqa: E402


def test_summarize_empty_db(tmp_path: Path) -> None:
    sm = SlowMemory(db_path=tmp_path / "empty.sqlite")
    try:
        s = sm.summarize()
        assert s == {
            "total": 0,
            "by_kind": {},
            "by_actor": {},
            "first_ts": None,
            "last_ts": None,
        }
    finally:
        sm.close()


def test_summarize_by_kind_counts(tmp_path: Path) -> None:
    sm = SlowMemory(db_path=tmp_path / "mixed.sqlite")
    try:
        sm.record("note", "curiosity", {"fact": "a"})
        sm.record("note", "curiosity", {"fact": "b"})
        sm.record("response", "reasoner-1", {"ok": True})
        sm.record("response", "reasoner-2", {"ok": True})
        sm.record("response", "reasoner-1", {"ok": False})
        sm.record("digest", "consolidator", {"n": 1})

        s = sm.summarize()
        assert s["total"] == 6
        assert s["by_kind"] == {"note": 2, "response": 3, "digest": 1}
        assert s["first_ts"] is not None
        assert s["last_ts"] is not None
        assert s["last_ts"] >= s["first_ts"]
    finally:
        sm.close()


def test_summarize_by_actor_aggregation(tmp_path: Path) -> None:
    sm = SlowMemory(db_path=tmp_path / "actors.sqlite")
    try:
        sm.record("response", "alice", {})
        sm.record("response", "alice", {})
        sm.record("response", "bob", {})
        sm.record("note", "alice", {"fact": "x"})
        s = sm.summarize()
        assert s["by_actor"] == {"alice": 3, "bob": 1}
    finally:
        sm.close()


def test_summarize_since_filter_excludes_older(tmp_path: Path) -> None:
    sm = SlowMemory(db_path=tmp_path / "since.sqlite")
    try:
        sm.record("note", "curiosity", {"fact": "old-1"})
        sm.record("note", "curiosity", {"fact": "old-2"})
        # Ensure a measurable gap so ts comparison is stable.
        time.sleep(0.05)
        cutoff = time.time()
        time.sleep(0.05)
        sm.record("response", "reasoner-1", {"ok": True})
        sm.record("response", "reasoner-2", {"ok": True})

        s_all = sm.summarize()
        assert s_all["total"] == 4

        s_recent = sm.summarize(since=cutoff)
        assert s_recent["total"] == 2
        assert s_recent["by_kind"] == {"response": 2}
        assert "note" not in s_recent["by_kind"]
    finally:
        sm.close()


def test_summarize_kind_filter(tmp_path: Path) -> None:
    sm = SlowMemory(db_path=tmp_path / "kindfilter.sqlite")
    try:
        sm.record("note", "curiosity", {"fact": "a"})
        sm.record("response", "r1", {})
        sm.record("response", "r2", {})
        s = sm.summarize(kind="response")
        assert s["total"] == 2
        assert s["by_kind"] == {"response": 2}
        assert set(s["by_actor"].keys()) == {"r1", "r2"}
    finally:
        sm.close()
