"""Tests for the SelfModel surface.

describe() returns the structural snapshot (reasoners, tools, latent,
recent_digest). answer() routes keyword queries to grounded descriptions.
"""
from __future__ import annotations

from pathlib import Path

from mente.memory import SlowMemory
from mente.reasoners import DeepSimulatedReasoner, FastHeuristicReasoner
from mente.self_model import SelfModel
from mente.state import LatentState
from mente.tools import ToolRegistry


def _build(tmp_path: Path) -> SelfModel:
    latent = LatentState(path=tmp_path / "latent.json")
    slow = SlowMemory(db_path=tmp_path / "episodic.sqlite")
    tools = ToolRegistry()

    @tools.register("clock.now", "Current time.", returns="str", est_cost_ms=0.5)
    async def _clock_now() -> str:
        return "2026-04-21T00:00:00"

    reasoners = [FastHeuristicReasoner(), DeepSimulatedReasoner()]
    return SelfModel(latent=latent, slow_mem=slow, reasoners=reasoners, tools=tools)


def test_describe_returns_reasoners_tools_latent_and_digest(tmp_path: Path) -> None:
    sm = _build(tmp_path)
    try:
        sm.latent.update(turns=7, last_reasoner="fast.heuristic")
        sm.latent.set("last_digest", {"total_responses": 1, "accept_rate": 1.0, "by_reasoner": {"fast.heuristic": 1}})
        d = sm.describe()

        names = [r["name"] for r in d["reasoners"]]
        assert "fast.heuristic" in names and "deep.sim" in names

        tools = [t["name"] for t in d["tools"]]
        assert "clock.now" in tools

        assert d["latent"]["turns"] == 7
        assert d["recent_digest"]["total_responses"] == 1
    finally:
        sm.slow_mem.close()


def test_answer_reasoner_query(tmp_path: Path) -> None:
    sm = _build(tmp_path)
    try:
        reply = sm.answer("what are your reasoners?")
        assert "reasoners" in reply.lower()
        assert "fast.heuristic" in reply
        assert "deep.sim" in reply
    finally:
        sm.slow_mem.close()


def test_answer_tool_query(tmp_path: Path) -> None:
    sm = _build(tmp_path)
    try:
        reply = sm.answer("what tools do you have?")
        assert "tool" in reply.lower()
        assert "clock.now" in reply
    finally:
        sm.slow_mem.close()


def test_answer_turn_query(tmp_path: Path) -> None:
    sm = _build(tmp_path)
    try:
        sm.latent.set("turns", 12)
        reply = sm.answer("how many turns have you done?")
        assert "12" in reply
    finally:
        sm.slow_mem.close()


def test_answer_doing_query_without_digest(tmp_path: Path) -> None:
    sm = _build(tmp_path)
    try:
        reply = sm.answer("what have you been doing?")
        # No digest recorded → falls back to a specific message.
        assert "no consolidation" in reply.lower() or "nothing" in reply.lower()
    finally:
        sm.slow_mem.close()


def test_answer_doing_query_with_digest(tmp_path: Path) -> None:
    sm = _build(tmp_path)
    try:
        sm.latent.set("last_digest", {
            "total_responses": 4,
            "accept_rate": 0.75,
            "by_reasoner": {"fast.heuristic": 3, "deep.sim": 1},
        })
        reply = sm.answer("what have you been doing?")
        assert "4" in reply
        assert "0.75" in reply
        assert "fast.heuristic" in reply
    finally:
        sm.slow_mem.close()


def test_answer_fallback_for_unknown_query(tmp_path: Path) -> None:
    sm = _build(tmp_path)
    try:
        reply = sm.answer("tell me a poem about the ocean")
        assert "I can answer" in reply or "reasoners" in reply.lower()
    finally:
        sm.slow_mem.close()
