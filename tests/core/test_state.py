"""Tests for aria.state.LatentState — disk-backed dict with checkpointing."""
from __future__ import annotations

import json
from pathlib import Path

from aria.state import LatentState


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
    assert data == {"turn": 3, "focus": "planning"}


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
