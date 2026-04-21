"""Tests for mente.memory: FastMemory TTL + SlowMemory SQLite episodic log."""
from __future__ import annotations

import time
from pathlib import Path

from mente.memory import FastMemory, SlowMemory


# --- FastMemory -------------------------------------------------------------

def test_fast_memory_set_and_get():
    m = FastMemory()
    m.set("k", "v")
    assert m.get("k") == "v"


def test_fast_memory_get_missing_returns_none():
    m = FastMemory()
    assert m.get("missing") is None


def test_fast_memory_permanent_entry_survives():
    m = FastMemory()
    m.set("k", 1)  # no TTL
    # Simulate elapsed time: since ttl is None, it's permanent.
    assert m.get("k") == 1


def test_fast_memory_ttl_expiration():
    m = FastMemory()
    m.set("k", "v", ttl_s=0.01)
    time.sleep(0.02)
    assert m.get("k") is None


def test_fast_memory_ttl_not_yet_expired():
    m = FastMemory()
    m.set("k", "v", ttl_s=60.0)
    assert m.get("k") == "v"


def test_fast_memory_all_skips_expired():
    m = FastMemory()
    m.set("alive", 1, ttl_s=60.0)
    m.set("dead", 2, ttl_s=0.01)
    m.set("forever", 3)
    time.sleep(0.02)
    snap = m.all()
    assert "alive" in snap
    assert "forever" in snap
    assert "dead" not in snap


def test_fast_memory_overwrite_resets_ts():
    m = FastMemory()
    m.set("k", "v1", ttl_s=0.01)
    time.sleep(0.02)
    # Now it'd be expired, but setting fresh restarts the clock.
    m.set("k", "v2", ttl_s=60.0)
    assert m.get("k") == "v2"


# --- SlowMemory -------------------------------------------------------------

def test_slow_memory_creates_db_file(tmp_root: Path):
    db = tmp_root / "sub" / "slow.db"
    mem = SlowMemory(db_path=db)
    try:
        assert db.exists()
    finally:
        mem.close()


def test_slow_memory_record_and_query_all(tmp_root: Path):
    db = tmp_root / "slow.db"
    mem = SlowMemory(db_path=db)
    try:
        mem.record("event", "actor1", {"k": "v"}, trace_id="tr_1")
        rows = mem.query()
        assert len(rows) == 1
        assert rows[0]["kind"] == "event"
        assert rows[0]["actor"] == "actor1"
        assert rows[0]["payload"] == {"k": "v"}
        assert rows[0]["trace_id"] == "tr_1"
        assert isinstance(rows[0]["ts"], float)
    finally:
        mem.close()


def test_slow_memory_query_filters_by_kind(tmp_root: Path):
    db = tmp_root / "slow.db"
    mem = SlowMemory(db_path=db)
    try:
        mem.record("a", "x", {"n": 1})
        mem.record("b", "x", {"n": 2})
        mem.record("a", "x", {"n": 3})
        rows = mem.query(kind="a")
        assert len(rows) == 2
        assert all(r["kind"] == "a" for r in rows)
    finally:
        mem.close()


def test_slow_memory_query_filters_by_since(tmp_root: Path):
    db = tmp_root / "slow.db"
    mem = SlowMemory(db_path=db)
    try:
        mem.record("a", "x", {"n": 1})
        time.sleep(0.01)
        cutoff = time.time()
        time.sleep(0.01)
        mem.record("a", "x", {"n": 2})
        rows = mem.query(since=cutoff)
        assert len(rows) == 1
        assert rows[0]["payload"] == {"n": 2}
    finally:
        mem.close()


def test_slow_memory_query_respects_limit(tmp_root: Path):
    db = tmp_root / "slow.db"
    mem = SlowMemory(db_path=db)
    try:
        for i in range(5):
            mem.record("k", "a", {"i": i})
        rows = mem.query(limit=3)
        assert len(rows) == 3
    finally:
        mem.close()


def test_slow_memory_query_orders_desc_by_ts(tmp_root: Path):
    db = tmp_root / "slow.db"
    mem = SlowMemory(db_path=db)
    try:
        mem.record("k", "a", {"i": 1})
        time.sleep(0.005)
        mem.record("k", "a", {"i": 2})
        time.sleep(0.005)
        mem.record("k", "a", {"i": 3})
        rows = mem.query()
        assert [r["payload"]["i"] for r in rows] == [3, 2, 1]
    finally:
        mem.close()


def test_slow_memory_schema_is_idempotent(tmp_root: Path):
    db = tmp_root / "slow.db"
    # Open, close, reopen — should not raise and should preserve rows.
    m1 = SlowMemory(db_path=db)
    m1.record("k", "a", {"n": 1}, trace_id="t1")
    m1.close()

    m2 = SlowMemory(db_path=db)
    try:
        rows = m2.query()
        assert len(rows) == 1
        assert rows[0]["trace_id"] == "t1"
    finally:
        m2.close()


def test_slow_memory_default_trace_id_is_empty(tmp_root: Path):
    db = tmp_root / "slow.db"
    mem = SlowMemory(db_path=db)
    try:
        mem.record("k", "a", {})
        rows = mem.query()
        assert rows[0]["trace_id"] == ""
    finally:
        mem.close()


def test_slow_memory_payload_roundtrips_nested(tmp_root: Path):
    db = tmp_root / "slow.db"
    mem = SlowMemory(db_path=db)
    try:
        nested = {"list": [1, 2, 3], "dict": {"a": "b"}}
        mem.record("k", "a", nested)
        rows = mem.query()
        assert rows[0]["payload"] == nested
    finally:
        mem.close()
