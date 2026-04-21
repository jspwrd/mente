"""Network helpers for TCP transport tests.

Picks an unused ephemeral port on 127.0.0.1 so multiple test units can run in
parallel without colliding on fixed ports.
"""
from __future__ import annotations

import socket


def find_unused_port(start: int = 7901) -> int:
    """Return an unused TCP port on 127.0.0.1.

    Binds to port 0 and asks the kernel for an ephemeral port. The ``start``
    value is informational; callers can use it when a deterministic search
    range is desired, but the default strategy is to let the kernel pick.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
