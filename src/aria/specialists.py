"""Legacy shim for specialist imports.

Kept for backwards compatibility with consumers that did
`from aria.specialists import MathSpecialist`. The canonical location is now
the `aria.specialists` package (see `specialists/__init__.py`), and Python's
import system prefers the package over this module when both are present.
This file is retained as documentation of the legacy surface.
"""
from __future__ import annotations

from .specialists.code import CodeSpecialist
from .specialists.math import MathSpecialist

__all__ = ["MathSpecialist", "CodeSpecialist"]
