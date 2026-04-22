"""Tests for mente.reasoners — FastHeuristicReasoner + DeepSimulatedReasoner."""
from __future__ import annotations

import pytest
from fixtures.cognition_helpers import make_world, register_default_tools

from mente.reasoners import (
    DeepSimulatedReasoner,
    FastHeuristicReasoner,
    set_self_model_hook,
)
from mente.tools import ToolRegistry
from mente.types import Belief, Intent


async def _setup(beliefs=None):
    world = await make_world(beliefs)
    tools = ToolRegistry()
    state = register_default_tools(tools)
    return world, tools, state


async def test_fast_hello_matches():
    world, tools, _ = await _setup()
    r = FastHeuristicReasoner()
    resp = await r.answer(Intent(text="hello there"), world, tools)
    assert "online" in resp.text.lower()
    assert resp.confidence >= 0.9
    assert resp.tier == "fast"


async def test_fast_time_uses_clock_tool():
    world, tools, _ = await _setup()
    r = FastHeuristicReasoner()
    resp = await r.answer(Intent(text="what time is it?"), world, tools)
    assert "2026-04-21" in resp.text
    assert "clock.now" in resp.tools_used


async def test_fast_remember_captures_fact():
    world, tools, state = await _setup()
    r = FastHeuristicReasoner()
    resp = await r.answer(Intent(text="remember that the sky is blue"), world, tools)
    assert "sky is blue" in resp.text
    assert state["notes"] == ["the sky is blue"]
    assert "memory.note" in resp.tools_used


async def test_fast_recall_lists_notes():
    world, tools, state = await _setup()
    state["notes"].extend(["a fact", "another fact"])
    r = FastHeuristicReasoner()
    resp = await r.answer(Intent(text="what do you remember?"), world, tools)
    assert "a fact" in resp.text
    assert "another fact" in resp.text
    assert "memory.recall" in resp.tools_used


async def test_fast_semantic_search_passthrough_hits():
    world, tools, state = await _setup()
    state["hits"].extend([
        {"text": "redis uses AOF", "score": 0.88},
        {"text": "postgres WAL", "score": 0.42},
    ])
    r = FastHeuristicReasoner()
    resp = await r.answer(Intent(text="what do you know about redis"), world, tools)
    assert "redis uses AOF" in resp.text
    assert "memory.search" in resp.tools_used
    assert resp.confidence >= 0.8


async def test_fast_semantic_search_no_hits_low_confidence():
    world, tools, _ = await _setup()
    r = FastHeuristicReasoner()
    resp = await r.answer(Intent(text="what do you know about quantum chromodynamics"), world, tools)
    assert "nothing relevant" in resp.text
    assert resp.confidence < 0.5


async def test_fast_who_am_i_with_and_without_belief():
    world, tools, _ = await _setup()
    r = FastHeuristicReasoner()
    resp = await r.answer(Intent(text="who am I?"), world, tools)
    assert resp.confidence < 0.5
    assert "someone" in resp.text.lower()

    world2, tools2, _ = await _setup([Belief(entity="user", attribute="name", value="Jasper")])
    resp2 = await r.answer(Intent(text="who am I?"), world2, tools2)
    assert "Jasper" in resp2.text
    assert resp2.confidence >= 0.9


async def test_fast_unknown_pattern_returns_zero_confidence():
    world, tools, _ = await _setup()
    r = FastHeuristicReasoner()
    resp = await r.answer(Intent(text="please derive the riemann hypothesis"), world, tools)
    assert resp.text == ""
    assert resp.confidence == 0.0


async def test_self_model_hook_invoked_for_self_query():
    world, tools, _ = await _setup()
    seen: list[str] = []

    def hook(text: str) -> str:
        seen.append(text)
        return "I am MENTE."

    set_self_model_hook(hook)
    try:
        r = FastHeuristicReasoner()
        resp = await r.answer(Intent(text="what are you?"), world, tools)
        assert seen == ["what are you?"]
        assert resp.text == "I am MENTE."
        assert resp.confidence >= 0.9
    finally:
        set_self_model_hook(None)


async def test_self_model_hook_absent_returns_fallback_text():
    world, tools, _ = await _setup()
    set_self_model_hook(None)
    r = FastHeuristicReasoner()
    resp = await r.answer(Intent(text="describe yourself"), world, tools)
    assert "self-model not attached" in resp.text


async def test_deep_simulated_reasoner_confidence_and_tier():
    world, tools, _ = await _setup()
    r = DeepSimulatedReasoner(est_cost_ms=5.0)  # small to keep test fast
    resp = await r.answer(Intent(text="ponder life"), world, tools)
    assert resp.tier == "deep"
    assert resp.confidence == pytest.approx(0.55)
    assert resp.reasoner == "deep.sim"


async def test_deep_simulated_reasoner_cites_matching_entity():
    world, tools, _ = await _setup([
        Belief(entity="redis", attribute="persistence", value="AOF"),
        Belief(entity="redis", attribute="kind", value="cache"),
    ])
    r = DeepSimulatedReasoner(est_cost_ms=5.0)
    resp = await r.answer(
        Intent(text="if I gave you three raspberry pis, could you run redis?"),
        world,
        tools,
    )
    assert "redis" in resp.text.lower()
    # Attribute surfaced so the caller sees what the deep fallback "knew".
    assert "AOF" in resp.text
    assert "persistence" in resp.text


async def test_deep_simulated_reasoner_invokes_memory_search_and_cites_top_hit():
    world, tools, state = await _setup()
    state["hits"].extend([
        {"text": "redis uses AOF", "score": 0.91},
        {"text": "postgres WAL", "score": 0.40},
    ])
    r = DeepSimulatedReasoner(est_cost_ms=5.0)
    resp = await r.answer(Intent(text="what do you know about redis"), world, tools)
    assert "redis uses AOF" in resp.text
    assert "memory.search" in resp.tools_used
    # Score formatted with two decimals so the reply feels cited, not guessed.
    assert "0.91" in resp.text


async def test_deep_simulated_reasoner_handles_empty_world_gracefully():
    world, tools, _ = await _setup()
    r = DeepSimulatedReasoner(est_cost_ms=5.0)
    resp = await r.answer(Intent(text="ponder something obscure"), world, tools)
    # No crash, no entity citation, install hint still appended.
    assert resp.confidence == pytest.approx(0.55)
    assert "deep-sim fallback" in resp.text


async def test_deep_simulated_reasoner_always_appends_install_hint():
    world, tools, state = await _setup([Belief(entity="redis", attribute="kind", value="cache")])
    state["hits"].append({"text": "redis uses AOF", "score": 0.9})
    r = DeepSimulatedReasoner(est_cost_ms=5.0)

    # Empty world / no tool match
    world_empty, tools_empty, _ = await _setup()
    resp_empty = await r.answer(Intent(text="just vibes"), world_empty, tools_empty)
    assert "deep-sim fallback" in resp_empty.text
    assert "mente[llm-ollama]" in resp_empty.text
    assert "ANTHROPIC_API_KEY" in resp_empty.text

    # Entity-match path
    resp_entity = await r.answer(Intent(text="tell me about redis"), world, tools)
    assert "deep-sim fallback" in resp_entity.text

    # Memory-search path
    resp_search = await r.answer(Intent(text="what do you know about redis"), world, tools)
    assert "deep-sim fallback" in resp_search.text


async def test_deep_simulated_reasoner_invokes_clock_on_time_intents():
    world, tools, _ = await _setup()
    r = DeepSimulatedReasoner(est_cost_ms=5.0)
    resp = await r.answer(Intent(text="when are you planning to sync?"), world, tools)
    assert "2026-04-21" in resp.text
    assert "clock.now" in resp.tools_used


async def test_deep_simulated_reasoner_tolerates_malformed_memory_hits():
    world, tools, state = await _setup()
    # Malformed entries: missing 'score' + non-dict value. Must not raise.
    state["hits"].extend([{"text": "partial"}, "not a dict"])
    r = DeepSimulatedReasoner(est_cost_ms=5.0)
    resp = await r.answer(Intent(text="what do you know about anything"), world, tools)
    assert resp.confidence == pytest.approx(0.55)
    assert "deep-sim fallback" in resp.text
