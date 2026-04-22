"""Consolidation loop — the 'sleep' step.

Periodically scans episodic memory, extracts patterns, and writes distilled
'digest' entries. Also produces a rolling self-summary of what the system
has been doing (feeding §9 Self-Model).

Phase 1: statistical summaries (counts, routing mix, verdict distribution,
most-noted facts). Deterministic and cheap.

Phase 2: an LLM-backed consolidator that writes natural-language summaries,
promotes stable facts into permanent world-model beliefs, and generates
training signal (good/bad turns) for router and verifier fine-tuning.
"""
from __future__ import annotations

import asyncio
import sqlite3
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .config import MenteConfig
from .logging import get_logger
from .memory import SlowMemory
from .state import LatentState

_log = get_logger("consolidator")


@dataclass
class Consolidator:
    slow_mem: SlowMemory
    latent: LatentState
    interval_s: float = 10.0
    config: MenteConfig | None = None

    def __post_init__(self) -> None:
        # Accept an optional MenteConfig so integrators using Consolidator on
        # its own (outside Runtime) still pick up env-driven cadence. Explicit
        # constructor kwargs remain authoritative.
        if self.config is not None and self.interval_s == 10.0:
            self.interval_s = self.config.consolidator_interval_s

    def digest(self) -> dict[str, Any]:
        rows = self.slow_mem.query(limit=500)
        responses = [r for r in rows if r.get("kind") == "response"]
        notes = [r for r in rows if r.get("kind") == "note"]
        by_reasoner = Counter(r.get("actor", "") for r in responses)
        verdicts = [
            r["payload"].get("verdict", {}).get("accept", True)
            for r in responses
            if isinstance(r.get("payload"), dict) and "verdict" in r["payload"]
        ]
        accept_rate = (
            sum(1 for v in verdicts if v) / len(verdicts)
            if len(verdicts) > 0
            else 1.0
        )
        avg_conf = (
            sum(
                r["payload"].get("verdict", {}).get("score", 0)
                for r in responses
                if isinstance(r.get("payload"), dict)
            ) / len(responses)
            if len(responses) > 0
            else 0.0
        )
        try:
            summary = self.slow_mem.summarize()
        except (sqlite3.Error, TypeError) as e:  # pragma: no cover - defensive
            _log.warning("summarize failed: %s", e)
            summary = {
                "total": 0,
                "by_kind": {},
                "by_actor": {},
                "first_ts": None,
                "last_ts": None,
            }
        return {
            "total_responses": len(responses),
            "by_reasoner": dict(by_reasoner),
            "accept_rate": round(accept_rate, 3),
            "avg_verdict_score": round(avg_conf, 3),
            "note_count": len(notes),
            "recent_notes": [
                n["payload"].get("fact")
                for n in notes[:10]
                if isinstance(n.get("payload"), dict)
            ],
            "summary": summary,
        }

    def consolidate(self) -> dict[str, Any]:
        try:
            digest = self.digest()
        except (sqlite3.Error, TypeError, RuntimeError) as e:
            _log.warning("consolidate failed: %s", e, exc_info=True)
            prior = self.latent.get("last_digest")
            if isinstance(prior, dict):
                return prior
            return {"error": str(e)}
        self.slow_mem.record("digest", "consolidator", digest)
        self.latent.set("last_digest", digest)
        self.latent.checkpoint()
        return digest

    async def run(self, stop: asyncio.Event) -> None:
        """Background task: consolidate every interval_s until stop is set.

        ``asyncio.CancelledError`` is not in the narrow catch list below so
        cooperative cancellation propagates and the task ends cleanly.
        """
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.interval_s)
            except TimeoutError:
                try:
                    self.consolidate()
                except (sqlite3.Error, RuntimeError, TypeError) as e:  # pragma: no cover
                    _log.warning("run-loop caught: %s", e, exc_info=True)
