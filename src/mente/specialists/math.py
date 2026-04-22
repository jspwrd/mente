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
from collections.abc import Callable
from dataclasses import dataclass

from ..tools import ToolRegistry
from ..types import Intent, ReasonerTier, Response
from ..world_model import WorldModel

_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Guard against CPU-bomb expressions like ``2 ** 999999999``. Pow with a
# large integer exponent is pure-Python big-integer math and can lock up
# the event loop — cap the exponent at a conservative level that still
# handles every reasonable arithmetic query.
_MAX_POW_EXPONENT = 1_000


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value) if isinstance(node.value, float) else node.value
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type in _BIN_OPS:
            left = _safe_eval(node.left)
            right = _safe_eval(node.right)
            if op_type is ast.Pow and abs(right) > _MAX_POW_EXPONENT:
                raise ValueError(f"exponent {right} exceeds safe limit {_MAX_POW_EXPONENT}")
            return _BIN_OPS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type_u = type(node.op)
        if op_type_u in _UNARY_OPS:
            return _UNARY_OPS[op_type_u](_safe_eval(node.operand))
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
