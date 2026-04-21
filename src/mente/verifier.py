"""Back-compat shim for the verifier module.

The implementation moved to ``mente.verifiers``. This file preserves the
legacy import path so ``from mente.verifier import Verifier, Verdict`` keeps
working for the Router/Runtime and any external code.
"""
from __future__ import annotations

from .verifiers import CompositeVerifier, HeuristicVerifier, StructuredVerifier, Verdict

# ``Verifier`` is the default implementation. Runtime constructs it as
# ``Verifier()`` so the default-constructor path must still work.
Verifier = HeuristicVerifier

__all__ = [
    "CompositeVerifier",
    "HeuristicVerifier",
    "StructuredVerifier",
    "Verdict",
    "Verifier",
]
