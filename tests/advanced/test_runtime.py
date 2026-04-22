"""Runtime-level integration tests.

Covers the lifecycle (start/shutdown), the fast→router→verifier→persist path
through handle_intent, the default tool set, the self-model wiring, and the
background consolidator/curiosity loops started by start_background.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fixtures.advanced_helpers import make_runtime, shutdown_runtime

from mente.runtime import Runtime
from mente.synthesis import LibraryStore, Primitive
from mente.types import Event, Intent


@pytest.mark.asyncio
async def test_start_and_shutdown_leaves_state_on_disk(tmp_path: Path) -> None:
    rt = await make_runtime(tmp_path)
    try:
        await rt.handle_intent(Intent(text="hello"))
    finally:
        await shutdown_runtime(rt)

    # After shutdown, SQLite + latent.json should exist and be readable.
    assert (rt.root / "episodic.sqlite").exists()
    assert (rt.root / "semantic.sqlite").exists()
    assert (rt.root / "latent.json").exists()
    data = json.loads((rt.root / "latent.json").read_text())
    # latent.json uses the versioned envelope; keys live under "values".
    assert data.get("_schema") == 1
    values = data.get("values", {})
    assert values.get("turns", 0) >= 1
    assert values.get("last_intent") == "hello"


@pytest.mark.asyncio
async def test_handle_intent_runs_fast_router_verifier_persist_path(tmp_path: Path) -> None:
    rt = await make_runtime(tmp_path)
    try:
        response_events: list[Event] = []
        intent_events: list[Event] = []

        async def on_resp(e: Event) -> None:
            response_events.append(e)

        async def on_intent(e: Event) -> None:
            intent_events.append(e)

        rt.bus.subscribe("response.*", on_resp, name="test.resp")
        rt.bus.subscribe("intent.*", on_intent, name="test.intent")

        response = await rt.handle_intent(Intent(text="hello"))

        # Fast-tier handled it, and the verifier stamp is present on the bus.
        assert response.text
        assert response.confidence >= 0.7
        await asyncio.sleep(0)  # let pending tasks flush

        assert any(e.topic.startswith("intent.") for e in intent_events)
        assert response_events, "runtime should emit at least one response.* event"
        resp_event = response_events[-1]
        assert "verdict" in resp_event.payload
        assert "accept" in resp_event.payload["verdict"]

        # Persistence subscribers should have written the response row too.
        rows = rt.slow_mem.query(kind="response", limit=5)
        assert rows, "response events should persist into slow_mem"

        # The latent state tracks turn + last reasoner.
        assert rt.latent.get("turns", 0) >= 1
        assert rt.latent.get("last_reasoner") is not None
    finally:
        await shutdown_runtime(rt)


@pytest.mark.asyncio
async def test_default_tools_registered(tmp_path: Path) -> None:
    rt = await make_runtime(tmp_path)
    try:
        names = {t.name for t in rt.tools.list()}
        assert {"clock.now", "memory.note", "memory.recall", "memory.search"} <= names

        now = await rt.tools.invoke("clock.now")
        assert now.ok and isinstance(now.value, str) and "T" in now.value

        saved = await rt.tools.invoke("memory.note", fact="mente uses an event bus")
        assert saved.ok and saved.value is True

        recalled = await rt.tools.invoke("memory.recall")
        assert recalled.ok and "mente uses an event bus" in recalled.value

        hits = await rt.tools.invoke("memory.search", query="event bus", k=2)
        assert hits.ok and isinstance(hits.value, list)
    finally:
        await shutdown_runtime(rt)


@pytest.mark.asyncio
async def test_self_model_hook_wired(tmp_path: Path) -> None:
    rt = await make_runtime(tmp_path)
    try:
        # The fast heuristic's "what are you / reasoners / tools" pattern triggers
        # the self-model hook. If the hook wasn't attached we'd get the fallback
        # string "self-model not attached".
        r = await rt.handle_intent(Intent(text="what are you"))
        assert "not attached" not in r.text
        # Should mention either reasoners, tools, or turns.
        assert any(k in r.text.lower() for k in ("reasoner", "tool", "turn", "digest", "activity"))
    finally:
        await shutdown_runtime(rt)


@pytest.mark.asyncio
async def test_library_store_loaded_from_disk(tmp_path: Path) -> None:
    # Pre-populate library.json before Runtime starts, so we exercise the
    # "load existing primitives on construction" path.
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    lib_path = state_dir / "library.json"
    primitive = {
        "name": "lib.fib.abc123",
        "source": "def fib(n):\n    return n\n",
        "entrypoint": "fib",
        "signature": {"n": "int"},
        "invocations": 3,
    }
    # v1 envelope format: {_schema, primitives}.
    lib_path.write_text(json.dumps({
        "_schema": 1,
        "primitives": {primitive["name"]: primitive},
    }))

    rt = Runtime(root=state_dir)
    try:
        await rt.start()
        assert isinstance(rt.library, LibraryStore)
        loaded = rt.library.list()
        names = {p.name for p in loaded}
        assert "lib.fib.abc123" in names
        got = rt.library.get("lib.fib.abc123")
        assert isinstance(got, Primitive)
        assert got.invocations == 3
    finally:
        await shutdown_runtime(rt)


@pytest.mark.asyncio
async def test_start_background_starts_and_stops_cleanly(tmp_path: Path) -> None:
    rt = await make_runtime(tmp_path)
    # Speed up the loops so the test doesn't block.
    rt.consolidator.interval_s = 0.05
    rt.curiosity.interval_s = 0.05
    try:
        tasks = rt.start_background()
        assert len(tasks) == 2
        assert all(isinstance(t, asyncio.Task) for t in tasks)
        assert all(not t.done() for t in tasks)

        # Let the loops iterate a few times.
        await asyncio.sleep(0.2)

        rt.stop_background()
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=1.0)
        assert all(t.done() for t in tasks)
    finally:
        await shutdown_runtime(rt)


@pytest.mark.asyncio
async def test_bus_close_is_idempotent(tmp_path: Path) -> None:
    """Closing the bus after a full shutdown should be a no-op, not raise."""
    rt = await make_runtime(tmp_path)
    await rt.handle_intent(Intent(text="hello"))
    await rt.shutdown()
    await rt.bus.close()  # must not raise


# -- deep-tier auto-select -------------------------------------------------


def _cfg_with_tier(tier: str) -> object:
    """Return a ``MenteConfig`` with an explicit ``llm_tier`` override."""
    import dataclasses

    from mente.config import MenteConfig
    return dataclasses.replace(MenteConfig.default(), llm_tier=tier)


@pytest.mark.asyncio
async def test_runtime_forces_sim_tier(tmp_path: Path) -> None:
    rt = Runtime(root=tmp_path / "state", config=_cfg_with_tier("sim"))
    try:
        names = [r.name for r in rt.reasoners]
        assert "deep.sim" in names
    finally:
        await shutdown_runtime(rt)


def test_runtime_forced_anthropic_without_key_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit llm_tier='anthropic' with no API key must raise loudly."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="anthropic|ANTHROPIC"):
        Runtime(root=tmp_path / "state", config=_cfg_with_tier("anthropic"))


@pytest.mark.asyncio
async def test_runtime_auto_falls_back_to_sim_when_nothing_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto mode in a CI-like environment (no key, no ollama) picks sim."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # If unit 1 has landed, force its probe False so the test is deterministic.
    try:
        import mente.llm_ollama as _llm_ollama
        monkeypatch.setattr(_llm_ollama, "ollama_available", lambda *a, **kw: False)
    except ImportError:
        pass  # module absent -> runtime's defensive fallback returns False
    rt = Runtime(root=tmp_path / "state", config=_cfg_with_tier("auto"))
    try:
        names = [r.name for r in rt.reasoners]
        assert "deep.sim" in names
    finally:
        await shutdown_runtime(rt)


@pytest.mark.asyncio
async def test_runtime_auto_picks_ollama_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``ollama_available`` mocked True, auto mode chooses deep.ollama."""
    import sys
    import types

    class _FakeOllamaReasoner:
        name = "deep.ollama"
        tier = "deep"
        est_cost_ms = 900.0

        def __init__(self, *, url: str = "", model: str = "") -> None:
            self.url = url
            self.model = model

        async def answer(self, *a: object, **kw: object) -> object:  # pragma: no cover
            raise AssertionError("not called in this test")

    stub = types.ModuleType("mente.llm_ollama")
    stub.OllamaReasoner = _FakeOllamaReasoner  # type: ignore[attr-defined]
    stub.ollama_available = lambda *a, **kw: True  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mente.llm_ollama", stub)

    rt = Runtime(root=tmp_path / "state", config=_cfg_with_tier("auto"))
    try:
        names = [r.name for r in rt.reasoners]
        assert "deep.ollama" in names
        assert "deep.sim" not in names
    finally:
        await shutdown_runtime(rt)


def test_runtime_forced_ollama_without_module_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``llm_tier='ollama'`` with no ``llm_ollama`` module -> install hint."""
    import sys
    # Mask any pre-existing module (sys.modules[x] = None short-circuits import).
    monkeypatch.setitem(sys.modules, "mente.llm_ollama", None)
    with pytest.raises(RuntimeError, match=r"mente\[llm-ollama\] not installed"):
        Runtime(root=tmp_path / "state", config=_cfg_with_tier("ollama"))
