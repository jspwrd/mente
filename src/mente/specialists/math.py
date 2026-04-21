"""Math specialist — safe arithmetic evaluation.

Moved verbatim from the previous monolithic `specialists.py` so that
imports like `from mente.specialists import MathSpecialist` keep working
(the package `__init__` re-exports, and `mente/specialists.py` is now
a thin shim for legacy consumers).
"""
from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass

from ..tools import ToolRegistry
from ..types import Intent, ReasonerTier, Response
from ..world_model import WorldModel


_SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


_MATH_RE = re.compile(r"(?:what is|compute|calculate|evaluate|solve)\s+([-+*/().\s0-9]+?)(?:[?.!]|$)", re.I)


@dataclass
class MathSpecialist:
    name: str = "specialist.math"
    tier: ReasonerTier = "specialist"
    est_cost_ms: float = 5.0

    async def answer(
        self, intent: Intent, world: WorldModel, tools: ToolRegistry
    ) -> Response:
        m = _MATH_RE.search(intent.text)
        if not m:
            # No arithmetic shape we recognize — low confidence, let the
            # router fall back.
            return Response(
                text="", reasoner=self.name, tier=self.tier,
                confidence=0.0, cost_ms=self.est_cost_ms,
            )
        expr = m.group(1).strip()
        try:
            tree = ast.parse(expr, mode="eval")
            result = _safe_eval(tree)
        except Exception as e:
            return Response(
                text=f"[math specialist: could not evaluate '{expr}' ({e})]",
                reasoner=self.name, tier=self.tier,
                confidence=0.1, cost_ms=self.est_cost_ms,
            )
        return Response(
            text=f"{expr} = {result}",
            reasoner=self.name, tier=self.tier,
            confidence=0.99, cost_ms=self.est_cost_ms,
        )
