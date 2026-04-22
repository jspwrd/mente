"""Heterogeneous router.

Dispatches an intent to the cheapest reasoner expected to clear a
confidence threshold. Falls through tiers if the chosen reasoner returns
low confidence.

Phase 1: argmax over metacog estimates with a cost/confidence trade-off.
Phase 2: trained router that learns from verifier feedback — when the
verifier rejects a fast-tier answer, that's training signal for the router
to escalate earlier next time.
"""
from __future__ import annotations

from dataclasses import dataclass

from .logging import get_logger
from .metacog import Metacog
from .reasoners import Reasoner
from .tools import ToolRegistry
from .types import Decision, Intent, Response
from .world_model import WorldModel

_log = get_logger("router")


@dataclass
class Router:
    reasoners: list[Reasoner]
    metacog: Metacog
    min_confidence: float = 0.7
    # Cost-to-confidence exchange rate: how many ms we'll spend per unit of
    # extra predicted confidence. Higher = willing to spend more for certainty.
    ms_per_conf: float = 2000.0

    def decide(self, intent: Intent) -> Decision:
        estimates = self.metacog.estimate(intent)
        # Rank: predicted_confidence - cost_ms/ms_per_conf
        ranked = sorted(
            estimates,
            key=lambda e: e.predicted_confidence - (e.predicted_cost_ms / self.ms_per_conf),
            reverse=True,
        )
        pick = ranked[0]
        reasoner = next(r for r in self.reasoners if r.name == pick.reasoner)
        return Decision(
            tier=reasoner.tier,
            reasoner=pick.reasoner,
            reason=pick.rationale,
            predicted_cost_ms=pick.predicted_cost_ms,
            predicted_confidence=pick.predicted_confidence,
        )

    async def route(
        self, intent: Intent, world: WorldModel, tools: ToolRegistry
    ) -> tuple[Decision, Response, list[Decision]]:
        """Route with fallback escalation if response confidence is too low."""
        attempted: list[Decision] = []
        decision = self.decide(intent)
        attempted.append(decision)
        # Intent text is NOT logged (may carry PII); keep it at DEBUG only.
        _log.info(
            "dispatch reasoner=%s predicted_confidence=%.2f predicted_cost_ms=%.1f",
            decision.reasoner,
            decision.predicted_confidence,
            decision.predicted_cost_ms,
            extra={"trace_id": intent.trace_id},
        )
        reasoner = next(r for r in self.reasoners if r.name == decision.reasoner)
        response = await reasoner.answer(intent, world, tools)

        # Escalate if confidence is below threshold and a deeper tier exists.
        tier_order = {"fast": 0, "deep": 1, "specialist": 1}
        if response.confidence < self.min_confidence:
            current = tier_order[decision.tier]
            deeper = [r for r in self.reasoners if tier_order[r.tier] > current]
            if deeper:
                next_r = max(deeper, key=lambda r: tier_order[r.tier])
                escalation = Decision(
                    tier=next_r.tier,
                    reasoner=next_r.name,
                    reason=f"escalated from {decision.reasoner} (conf {response.confidence:.2f})",
                    predicted_cost_ms=next_r.est_cost_ms,
                    predicted_confidence=0.7,
                )
                attempted.append(escalation)
                _log.info(
                    "escalate from=%s to=%s response_confidence=%.2f",
                    decision.reasoner,
                    escalation.reasoner,
                    response.confidence,
                    extra={"trace_id": intent.trace_id},
                )
                response = await next_r.answer(intent, world, tools)
                decision = escalation

        return decision, response, attempted
