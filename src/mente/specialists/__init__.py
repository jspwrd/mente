"""Specialist reasoners package.

Standalone reasoners with narrow competence and higher confidence inside
their domain than a general model. In Phase 3 each specialist typically
lives in its own process, announced onto the bus via `meta.capability.*`.

Phase 1 includes:
  - MathSpecialist: evaluates arithmetic expressions safely
  - CodeSpecialist: critiques a Python snippet via AST inspection
"""
from __future__ import annotations

from .code import CodeSpecialist
from .math import MathSpecialist, _safe_eval

__all__ = ["MathSpecialist", "CodeSpecialist", "_safe_eval"]
