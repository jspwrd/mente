"""Tests for mente.state.LatentState — disk-backed dict with checkpointing."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from mente.state import LatentState


class _RecordCapture(logging.Handler):
    """Plain handler that records LogRecords into a list.

    ``caplog`` only captures via root, and ``mente.*`` has ``propagate=False``
    once ``configure()`` has been called, so we attach directly to the target
    logger in tests.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _capture(logger_name: str) -> tuple[logging.Logger, _RecordCapture]:
    logger = logging.getLogger(logger_name)
    cap = _RecordCapture()
    logger.addHandler(cap)
    prior = logger.level
    logger.setLevel(logging.DEBUG)
    cap._prior_level = prior  # type: ignore[attr-defined]
    return logger, cap


def _release(logger: logging.Logger, cap: _RecordCapture) -> None:
    logger.removeHandler(cap)
    logger.setLevel(cap._prior_level)  # type: ignore[attr-defined]


def test_load_missing_file_returns_empty(tmp_root: Path):
    path = tmp_root / "latent.json"
    s = LatentState.load(path)
    assert s.values == {}
    assert s.path == path


def test_set_and_get_roundtrip():
    s = LatentState()
    s.set("mood", "curious")
    assert s.get("mood") == "curious"


def test_get_default_when_missing():
    s = LatentState()
    assert s.get("absent") is None
    assert s.get("absent", "fallback") == "fallback"


def test_update_merges_values():
    s = LatentState()
    s.update(a=1, b=2)
    assert s.get("a") == 1
    assert s.get("b") == 2


def test_checkpoint_writes_file(tmp_root: Path):
    path = tmp_root / "state.json"
    s = LatentState(path=path)
    s.set("turn", 3)
    s.set("focus", "planning")
    s.checkpoint()
    assert path.exists()
    data = json.loads(path.read_text())
    # On-disk shape is the versioned envelope.
    assert data == {"_schema": 1, "values": {"turn": 3, "focus": "planning"}}


def test_checkpoint_then_reload_roundtrip(tmp_root: Path):
    path = tmp_root / "state.json"
    s = LatentState(path=path)
    s.set("counter", 7)
    s.set("nested", {"a": [1, 2, 3]})
    s.checkpoint()

    # Fresh instance reads what was written.
    s2 = LatentState.load(path)
    assert s2.get("counter") == 7
    assert s2.get("nested") == {"a": [1, 2, 3]}


def test_checkpoint_without_path_is_noop(tmp_root: Path):
    s = LatentState()  # path=None
    s.set("x", 1)
    # Should not raise.
    s.checkpoint()


def test_checkpoint_creates_parent_dirs(tmp_root: Path):
    nested = tmp_root / "deeply" / "nested" / "state.json"
    s = LatentState(path=nested)
    s.set("k", "v")
    s.checkpoint()
    assert nested.exists()


def test_checkpoint_is_atomic_replace(tmp_root: Path):
    path = tmp_root / "state.json"
    s = LatentState(path=path)
    s.set("v", 1)
    s.checkpoint()
    # After a successful checkpoint no .tmp sibling should remain.
    assert not (path.parent / (path.name + ".tmp")).exists()


def test_checkpoint_overwrites_previous_content(tmp_root: Path):
    path = tmp_root / "state.json"
    s = LatentState(path=path)
    s.set("v", "first")
    s.checkpoint()
    s.set("v", "second")
    s.checkpoint()

    s2 = LatentState.load(path)
    assert s2.get("v") == "second"


def test_set_overwrites_existing():
    s = LatentState()
    s.set("k", 1)
    s.set("k", 2)
    assert s.get("k") == 2


def test_load_after_checkpoint_preserves_path(tmp_root: Path):
    path = tmp_root / "s.json"
    s = LatentState(path=path)
    s.set("a", 1)
    s.checkpoint()
    s2 = LatentState.load(path)
    assert s2.path == path


# ---- schema versioning ----

def test_load_old_format_migrates_and_logs(tmp_root: Path):
    """A pre-versioning (v0) latent.json — a bare dict — migrates cleanly."""
    path = tmp_root / "latent.json"
    path.write_text(json.dumps({"counter": 4, "mood": "quiet"}))

    logger, cap = _capture("mente.state")
    try:
        s = LatentState.load(path)
    finally:
        _release(logger, cap)

    assert s.get("counter") == 4
    assert s.get("mood") == "quiet"
    assert any("pre-versioning" in r.getMessage() for r in cap.records)


def test_load_v1_roundtrip_preserves_content(tmp_root: Path):
    """Writing a v1 envelope and reloading preserves every key."""
    path = tmp_root / "latent.json"
    s = LatentState(path=path)
    s.set("n", 42)
    s.set("tree", {"leaf": [1, 2, 3]})
    s.checkpoint()

    s2 = LatentState.load(path)
    assert s2.values == {"n": 42, "tree": {"leaf": [1, 2, 3]}}


def test_load_unknown_future_version_starts_empty(tmp_root: Path):
    """A latent.json with a schema newer than we support logs + starts empty."""
    path = tmp_root / "latent.json"
    path.write_text(json.dumps({"_schema": 999, "values": {"n": 1}}))

    logger, cap = _capture("mente.state")
    try:
        s = LatentState.load(path)
    finally:
        _release(logger, cap)

    assert s.values == {}
    assert any("newer than supported" in r.getMessage() for r in cap.records)


def test_checkpoint_writes_schema_version(tmp_root: Path):
    """Every checkpoint writes the current ``_SCHEMA_VERSION`` envelope."""
    path = tmp_root / "latent.json"
    s = LatentState(path=path)
    s.set("k", "v")
    s.checkpoint()
    data = json.loads(path.read_text())
    assert data["_schema"] == 1
    assert data["values"] == {"k": "v"}
