"""Structured logging helpers for ARIA.

A thin layer over the stdlib ``logging`` module that adds:

* a single ``configure()`` entry point (idempotent)
* a JSON formatter for machine-parseable lines
* per-task context (e.g. ``trace_id``) via ``contextvars``
* best-effort redaction of API-key-like strings

No third-party dependencies. The module is named ``aria.logging`` which
shadows ``import logging`` inside this file, so we alias to ``stdlog``.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import logging as stdlog
import re
import sys
from typing import Any, Iterator, TextIO

__all__ = [
    "get_logger",
    "configure",
    "JsonFormatter",
    "bind",
    "redact_secrets",
]


# ---------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------

_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "aria_log_context", default={}
)


@contextlib.contextmanager
def bind(**ctx: Any) -> Iterator[None]:
    """Attach ``ctx`` to every log record emitted inside the ``with`` block.

    Uses ``contextvars`` so the context is isolated per asyncio task / thread
    that has its own context.
    """
    current = _context.get()
    merged = {**current, **ctx}
    token = _context.set(merged)
    try:
        yield
    finally:
        _context.reset(token)


# ---------------------------------------------------------------------------
# redaction
# ---------------------------------------------------------------------------

# Ordered most-specific first so e.g. ``sk-ant-`` wins over a generic fallback.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]+"),
    re.compile(r"pa-[A-Za-z0-9_\-]+"),
    re.compile(r"AKIA[A-Z0-9]{12,}"),
    re.compile(r"ghp_[A-Za-z0-9]+"),
    re.compile(r"ghs_[A-Za-z0-9]+"),
    re.compile(r"gho_[A-Za-z0-9]+"),
    re.compile(r"pat_[A-Za-z0-9]+"),
    re.compile(r"glpat-[A-Za-z0-9_\-]+"),
)


def redact_secrets(text: str) -> str:
    """Replace API-key-like substrings with a masked marker.

    Each match keeps its first 4 characters and is followed by
    ``***REDACTED***``. E.g. ``sk-ant-1234567890abc`` -> ``sk-a***REDACTED***``.
    """

    def _sub(match: re.Match[str]) -> str:
        s = match.group(0)
        prefix = s[:4]
        return f"{prefix}***REDACTED***"

    out = text
    for pat in _SECRET_PATTERNS:
        out = pat.sub(_sub, out)
    return out


# ---------------------------------------------------------------------------
# formatter
# ---------------------------------------------------------------------------


class JsonFormatter(stdlog.Formatter):
    """Emit each record as a single JSON object.

    Shape::

        {"ts":..., "level":..., "logger":..., "msg":..., **extras, **context}

    ``extras`` are any attributes attached to the record via ``extra=...`` on
    the logging call. ``context`` is whatever is currently bound via
    :func:`bind`.
    """

    # Attributes that ``LogRecord`` sets itself; we ignore them when looking
    # for user-supplied extras.
    _RESERVED: frozenset[str] = frozenset(
        {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
            "asctime", "taskName",
        }
    )

    def format(self, record: stdlog.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": record.created,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # extras (anything non-reserved on the record dict)
        for key, value in record.__dict__.items():
            if key in self._RESERVED or key.startswith("_"):
                continue
            if key in payload:
                continue
            payload[key] = value

        # currently bound context overrides colliding extras by design — the
        # active trace_id should win.
        for key, value in _context.get().items():
            payload[key] = value

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


class _ContextPlainFormatter(stdlog.Formatter):
    """Plain formatter that appends bound context as ``k=v`` pairs."""

    def format(self, record: stdlog.LogRecord) -> str:
        base = super().format(record)
        ctx = _context.get()
        if not ctx:
            return base
        suffix = " ".join(f"{k}={v}" for k, v in ctx.items())
        return f"{base} {suffix}"


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

_ROOT_NAME = "aria"
_HANDLER_NAME = "aria._default_handler"
_configured: bool = False


def configure(
    level: str = "INFO",
    json: bool = False,
    stream: TextIO = sys.stderr,
) -> None:
    """Configure the ``aria`` root logger once.

    Idempotent: calling more than once is a no-op (the handler is not
    re-installed, level is not changed). This makes it safe to call from
    multiple entry points.
    """
    global _configured
    if _configured:
        return

    logger = stdlog.getLogger(_ROOT_NAME)
    logger.setLevel(level)
    logger.propagate = False

    # Defensive: if a handler with our sentinel name already exists (e.g. from
    # a previous process-level setup), don't stack another on top.
    for existing in logger.handlers:
        if getattr(existing, "name", None) == _HANDLER_NAME:
            _configured = True
            return

    handler = stdlog.StreamHandler(stream)
    handler.name = _HANDLER_NAME
    handler.setLevel(level)

    formatter: stdlog.Formatter
    if json:
        formatter = JsonFormatter()
    else:
        formatter = _ContextPlainFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    _configured = True


def get_logger(name: str) -> stdlog.Logger:
    """Return a child logger under the ``aria`` tree.

    ``get_logger("runtime")`` -> logger named ``aria.runtime``. Bare ``aria``
    is returned if ``name`` is empty.
    """
    if not name:
        return stdlog.getLogger(_ROOT_NAME)
    return stdlog.getLogger(f"{_ROOT_NAME}.{name}")
