"""Library reload cycle tests.

Pins behavior for the on-disk ``LibraryStore`` round-trip:
  * invocation counters persist across distinct store instances sharing a path
  * a ``save()`` + fresh load preserves the primitive dict verbatim
  * corrupt / partial entries log + skip rather than brick runtime construction
  * pre-versioning (v0) files migrate into the v1 envelope
  * v1 files round-trip exactly
  * future-versioned files start empty (with a warning)
  * unknown payload fields are dropped so forward-compat downgrades work
"""
from __future__ import annotations

import json
import logging

from mente.bus import EventBus
from mente.synthesis import LibraryStore, Primitive, SynthesisReasoner
from mente.tools import ToolRegistry
from mente.types import Intent
from mente.world_model import WorldModel

_FIB_INTENT = "compute the 10th fibonacci number"


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


async def _synthesize_once(library: LibraryStore) -> str:
    """Run one synthesis pass through a fresh reasoner; return primitive name."""
    tools = ToolRegistry()
    reasoner = SynthesisReasoner(library=library, tools=tools)
    world = WorldModel(bus=EventBus())
    resp = await reasoner.answer(Intent(text=_FIB_INTENT), world, ToolRegistry())
    assert resp.tools_used, f"synthesis did not promote a primitive: {resp.text!r}"
    return resp.tools_used[0]


async def test_library_reuse_persists_across_reloads(tmp_path):
    lib_path = tmp_path / "library.json"

    # First cycle: cold store, fresh synthesis → invocations == 1.
    store_a = LibraryStore(path=lib_path)
    name = await _synthesize_once(store_a)
    assert store_a.get(name).invocations == 1

    # Second cycle: brand-new store over the same path, brand-new reasoner.
    # The stored primitive should be re-hydrated, and the repeat intent
    # should bump invocations to 2 without creating a duplicate entry.
    store_b = LibraryStore(path=lib_path)
    assert store_b.get(name) is not None, "primitive did not survive reload"
    name_b = await _synthesize_once(store_b)
    assert name_b == name
    assert store_b.get(name).invocations == 2
    assert len(store_b.list()) == 1

    # Disk file carries the incremented counter inside the v1 envelope.
    on_disk = json.loads(lib_path.read_text())
    assert on_disk["_schema"] == 1
    assert on_disk["primitives"][name]["invocations"] == 2


def test_library_file_survives_process_restart(tmp_path):
    lib_path = tmp_path / "library.json"

    store = LibraryStore(path=lib_path)
    store.add(Primitive(
        name="lib.fib.deadbe",
        source="def fib(n):\n    return n\n",
        entrypoint="fib",
        signature={"n": "int"},
        invocations=1,
    ))

    # Mutate via direct dict access, then persist.
    store._primitives["lib.fib.deadbe"].invocations = 42
    store.save()

    # Fresh load from the same path → round-trip integrity.
    reloaded = LibraryStore(path=lib_path)
    assert reloaded._primitives.keys() == store._primitives.keys()
    for key, prim in store._primitives.items():
        assert reloaded._primitives[key].__dict__ == prim.__dict__


def test_library_ignores_corrupt_entries_gracefully(tmp_path):
    lib_path = tmp_path / "library.json"
    # One valid entry + one missing the ``source`` required field — written
    # inside a v1 envelope.
    payload = {
        "_schema": 1,
        "primitives": {
            "lib.good": {
                "name": "lib.good",
                "source": "def good():\n    return 1\n",
                "entrypoint": "good",
                "signature": {},
                "invocations": 1,
            },
            "lib.bad": {
                "name": "lib.bad",
                # 'source' intentionally omitted
                "entrypoint": "bad",
                "signature": {},
                "invocations": 1,
            },
        },
    }
    lib_path.write_text(json.dumps(payload))

    # Resilient behavior: corrupt entries are logged + skipped so a user's
    # disk corruption can't brick runtime construction. Good entries load.
    store = LibraryStore(path=lib_path)
    assert store.get("lib.good") is not None
    assert store.get("lib.bad") is None


def test_library_migrates_from_preversioning_v0(tmp_path):
    """An old bare-dict ``library.json`` loads via v0→v1 migration."""
    lib_path = tmp_path / "library.json"
    # v0 shape: bare {name: payload} — no _schema key.
    legacy = {
        "lib.old": {
            "name": "lib.old",
            "source": "def old():\n    return 1\n",
            "entrypoint": "old",
            "signature": {},
            "invocations": 3,
        },
    }
    lib_path.write_text(json.dumps(legacy))

    logger, cap = _capture("mente.synthesis")
    try:
        store = LibraryStore(path=lib_path)
    finally:
        _release(logger, cap)

    assert store.get("lib.old") is not None
    assert store.get("lib.old").invocations == 3
    assert any("pre-versioning" in r.getMessage() for r in cap.records)

    # Next save rewrites in v1 envelope form.
    store.save()
    on_disk = json.loads(lib_path.read_text())
    assert on_disk["_schema"] == 1
    assert "primitives" in on_disk
    assert on_disk["primitives"]["lib.old"]["invocations"] == 3


def test_library_v1_roundtrip_preserves_content(tmp_path):
    """Writing a v1 file and reloading it preserves every primitive field."""
    lib_path = tmp_path / "library.json"
    store = LibraryStore(path=lib_path)
    store.add(Primitive(
        name="lib.a", source="def a(): return 1\n", entrypoint="a",
        signature={}, invocations=5,
    ))
    store.add(Primitive(
        name="lib.b", source="def b(x): return x\n", entrypoint="b",
        signature={"x": "int"}, invocations=7,
    ))

    reloaded = LibraryStore(path=lib_path)
    assert set(reloaded._primitives) == {"lib.a", "lib.b"}
    for key, prim in store._primitives.items():
        assert reloaded._primitives[key].__dict__ == prim.__dict__


def test_library_unknown_future_version_starts_empty(tmp_path):
    """A future-versioned file logs a warning and starts empty."""
    lib_path = tmp_path / "library.json"
    lib_path.write_text(json.dumps({
        "_schema": 999,
        "primitives": {"lib.future": {"name": "x", "source": "y",
                                      "entrypoint": "z", "signature": {}}},
    }))

    logger, cap = _capture("mente.synthesis")
    try:
        store = LibraryStore(path=lib_path)
    finally:
        _release(logger, cap)

    assert store.list() == []
    assert any("newer than supported" in r.getMessage() for r in cap.records)


def test_library_ignores_unknown_fields_in_payload(tmp_path):
    """A primitive payload with extra unknown fields loads (fields dropped)."""
    lib_path = tmp_path / "library.json"
    lib_path.write_text(json.dumps({
        "_schema": 1,
        "primitives": {
            "lib.ext": {
                "name": "lib.ext",
                "source": "def ext(): return 1\n",
                "entrypoint": "ext",
                "signature": {},
                "invocations": 2,
                # Fields a future schema might add — must be ignored cleanly.
                "author": "future-me",
                "tags": ["fast", "pure"],
            },
        },
    }))

    store = LibraryStore(path=lib_path)
    prim = store.get("lib.ext")
    assert prim is not None
    assert prim.invocations == 2
    # Unknown fields didn't leak into the dataclass.
    assert not hasattr(prim, "author")
    assert not hasattr(prim, "tags")


def test_library_missing_required_field_is_skipped(tmp_path):
    """A payload missing a required dataclass field is logged + skipped."""
    lib_path = tmp_path / "library.json"
    lib_path.write_text(json.dumps({
        "_schema": 1,
        "primitives": {
            "lib.good": {
                "name": "lib.good",
                "source": "def good():\n    return 1\n",
                "entrypoint": "good",
                "signature": {},
                "invocations": 1,
            },
            "lib.bad": {
                # 'source' and 'entrypoint' missing
                "name": "lib.bad",
                "signature": {},
                "invocations": 0,
            },
        },
    }))

    logger, cap = _capture("mente.synthesis")
    try:
        store = LibraryStore(path=lib_path)
    finally:
        _release(logger, cap)

    assert store.get("lib.good") is not None
    assert store.get("lib.bad") is None
    assert any("malformed" in r.getMessage() for r in cap.records)
