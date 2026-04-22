"""Back-compat shim for the verifier module.

The implementation moved to ``mente.verifiers``. This file preserves the
legacy import path so ``from mente.verifier import Verifier, Verdict`` keeps
working for the Router/Runtime and any external code.

The shim is also where we attach module-level logging: rejected verdicts
emit a WARNING so operators see them without having to crank the whole
package to DEBUG. We install the wrapper here (rather than in the
``mente.verifiers`` package) so the pure verifier implementation stays
free of logging concerns and the identity ``Verifier is HeuristicVerifier``
is preserved for back-compat.
"""
from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING

from .logging import get_logger
from .verifiers import CompositeVerifier, HeuristicVerifier, StructuredVerifier, Verdict

if TYPE_CHECKING:
    from .types import Intent, Response
    from .world_model import WorldModel

_log = get_logger("verifier")


def _install_reject_logging() -> None:
    """Wrap ``HeuristicVerifier.verify`` once to log rejected verdicts.

    Idempotent: marks the wrapped function with a sentinel attribute so a
    re-import (or test reload) won't stack wrappers.
    """
    original = HeuristicVerifier.verify
    if getattr(original, "_mente_reject_log_installed", False):
        return

    @wraps(original)
    def verify_with_logging(
        self: HeuristicVerifier,
        intent: Intent,
        response: Response,
        world: WorldModel,
    ) -> Verdict:
        verdict = original(self, intent, response, world)
        if not verdict.accept:
            _log.warning(
                "verdict rejected score=%.2f reasons=%s",
                verdict.score,
                verdict.reasons,
                extra={"trace_id": response.trace_id},
            )
        return verdict

    verify_with_logging._mente_reject_log_installed = True  # type: ignore[attr-defined]
    HeuristicVerifier.verify = verify_with_logging  # type: ignore[method-assign]


_install_reject_logging()

# ``Verifier`` is the default implementation. Runtime constructs it as
# ``Verifier()`` so the default-constructor path must still work. Identity
# with ``HeuristicVerifier`` is load-bearing (see back-compat tests).
Verifier = HeuristicVerifier

__all__ = [
    "CompositeVerifier",
    "HeuristicVerifier",
    "StructuredVerifier",
    "Verdict",
    "Verifier",
]
