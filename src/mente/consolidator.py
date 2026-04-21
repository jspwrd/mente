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
import sys
import traceback
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .memory import SlowMemory
from .state import LatentState


@dataclass
class Consolidator:
    slow_mem: SlowMemory
    latent: LatentState
    interval_s: float = 10.0

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
        except Exception as e:  # pragma: no cover - defensive
            print(f"[consolidator] summarize failed: {e}", file=sys.stderr)
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
            self.slow_mem.record("digest", "consolidator", digest)
            self.latent.set("last_digest", digest)
            self.latent.checkpoint()
            return digest
        except Exception as e:
            print(f"[consolidator] consolidate failed: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            prior = self.latent.get("last_digest")
            if isinstance(prior, dict):
                return prior
            return {"error": str(e)}

    async def run(self, stop: asyncio.Event) -> None:
        """Background task: consolidate every interval_s until stop is set."""
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.interval_s)
            except TimeoutError:
                try:
                    self.consolidate()
                except Exception as e:  # pragma: no cover - belt & suspenders
                    print(
                        f"[consolidator] run-loop caught unexpected: {e}",
                        file=sys.stderr,
                    )
                    traceback.print_exc(file=sys.stderr)
