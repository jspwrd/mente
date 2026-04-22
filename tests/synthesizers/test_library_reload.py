"""Library reload cycle tests.

Pins behavior for the on-disk ``LibraryStore`` round-trip:
  * invocation counters persist across distinct store instances sharing a path
  * a ``save()`` + fresh load preserves the primitive dict verbatim
  * a partially-corrupt ``library.json`` raises cleanly at load time (current
    behavior — no silent skip), so regressions to silent-drop surface here.
"""
from __future__ import annotations

import json

import pytest

from mente.bus import EventBus
from mente.synthesis import LibraryStore, Primitive, SynthesisReasoner
from mente.tools import ToolRegistry
from mente.types import Intent
from mente.world_model import WorldModel

_FIB_INTENT = "compute the 10th fibonacci number"


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

    # Disk file carries the incremented counter.
    on_disk = json.loads(lib_path.read_text())
    assert on_disk[name]["invocations"] == 2


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
    # One valid entry + one missing the ``source`` required field.
    payload = {
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
    }
    lib_path.write_text(json.dumps(payload))

    # Current behavior: ``__post_init__`` materializes entries via
    # ``Primitive(**v)``, so a missing required field surfaces as ``TypeError``.
    # Pinning this so any future switch to silent-skip is a deliberate change.
    with pytest.raises(TypeError):
        LibraryStore(path=lib_path)
