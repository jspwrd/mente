"""Tests for aria.logging."""
from __future__ import annotations

import asyncio
import io
import json
import logging as stdlog
import sys

import pytest

from aria import logging as aria_logging
from aria.logging import (
    JsonFormatter,
    bind,
    configure,
    get_logger,
    redact_secrets,
)


@pytest.fixture(autouse=True)
def _reset_logging_state():
    """Ensure each test starts from a clean configuration.

    The module guards ``configure()`` with a module-level flag and attaches a
    named handler to the ``aria`` logger. Tear both down between tests so
    assertions don't leak.
    """
    yield
    aria_logging._configured = False
    root = stdlog.getLogger("aria")
    for h in list(root.handlers):
        if getattr(h, "name", None) == "aria._default_handler":
            root.removeHandler(h)
    # Clear context var to the empty default.
    aria_logging._context.set({})


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------


def test_get_logger_returns_child_under_aria_tree():
    # Touch the root first so Python's logger tree materializes the parent.
    aria_root = stdlog.getLogger("aria")
    logger = get_logger("runtime")
    assert logger.name == "aria.runtime"
    assert logger.parent is aria_root


def test_get_logger_empty_name_returns_root():
    logger = get_logger("")
    assert logger.name == "aria"


# ---------------------------------------------------------------------------
# configure + JSON formatter
# ---------------------------------------------------------------------------


def test_configure_json_emits_expected_keys():
    stream = io.StringIO()
    configure(level="DEBUG", json=True, stream=stream)
    logger = get_logger("example")

    logger.info("hello world", extra={"user": "alice"})

    line = stream.getvalue().strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["msg"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "aria.example"
    assert "ts" in payload
    assert payload["user"] == "alice"


def test_configure_writes_json_to_stream():
    stream = io.StringIO()
    configure(level="INFO", json=True, stream=stream)
    logger = get_logger("stream")
    logger.info("line one")

    line = stream.getvalue().strip().splitlines()[0]
    payload = json.loads(line)
    assert payload["msg"] == "line one"
    assert payload["logger"] == "aria.stream"


def test_configure_twice_does_not_double_install_handlers():
    stream = io.StringIO()
    configure(level="INFO", json=True, stream=stream)
    configure(level="INFO", json=True, stream=stream)
    configure(level="DEBUG", json=False, stream=stream)

    root = stdlog.getLogger("aria")
    named = [
        h for h in root.handlers
        if getattr(h, "name", None) == "aria._default_handler"
    ]
    assert len(named) == 1


# ---------------------------------------------------------------------------
# bind / context
# ---------------------------------------------------------------------------


def test_bind_attaches_context_to_json_output():
    stream = io.StringIO()
    configure(level="INFO", json=True, stream=stream)
    logger = get_logger("ctx")

    with bind(trace_id="abc", request="r1"):
        logger.info("bound")

    line = stream.getvalue().strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["trace_id"] == "abc"
    assert payload["request"] == "r1"


def test_bind_is_scoped_to_block():
    stream = io.StringIO()
    configure(level="INFO", json=True, stream=stream)
    logger = get_logger("ctx")

    with bind(trace_id="abc"):
        logger.info("inside")
    logger.info("outside")

    lines = [
        json.loads(ln) for ln in stream.getvalue().strip().splitlines()
    ]
    inside = next(p for p in lines if p["msg"] == "inside")
    outside = next(p for p in lines if p["msg"] == "outside")
    assert inside["trace_id"] == "abc"
    assert "trace_id" not in outside


@pytest.mark.asyncio
async def test_bind_isolated_per_asyncio_task():
    """Each task gets its own copy of the ContextVar when created with
    ``asyncio.create_task`` — bindings in one task must not leak to another.
    """
    stream = io.StringIO()
    configure(level="INFO", json=True, stream=stream)
    logger = get_logger("async")

    started = asyncio.Event()
    release = asyncio.Event()

    async def worker_a() -> None:
        with bind(trace_id="task-a"):
            started.set()
            await release.wait()
            logger.info("from-a")

    async def worker_b() -> None:
        await started.wait()
        # No bind here; should not observe task-a's trace_id.
        logger.info("from-b-before")
        with bind(trace_id="task-b"):
            logger.info("from-b-bound")
        release.set()

    await asyncio.gather(worker_a(), worker_b())

    payloads = [
        json.loads(ln) for ln in stream.getvalue().strip().splitlines()
    ]
    by_msg = {p["msg"]: p for p in payloads}
    assert by_msg["from-a"]["trace_id"] == "task-a"
    assert "trace_id" not in by_msg["from-b-before"]
    assert by_msg["from-b-bound"]["trace_id"] == "task-b"


# ---------------------------------------------------------------------------
# redact_secrets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("sk-ant-1234567890abc", "sk-a***REDACTED***"),
        ("pa-abcdef123456", "pa-a***REDACTED***"),
        ("AKIAABCDEFGHIJKLMNOP", "AKIA***REDACTED***"),
        ("ghp_abcdef1234567890", "ghp_***REDACTED***"),
        ("ghs_abcdef1234567890", "ghs_***REDACTED***"),
        ("gho_abcdef1234567890", "gho_***REDACTED***"),
        ("pat_abcdef1234567890", "pat_***REDACTED***"),
        ("glpat-abcdef1234567890", "glpa***REDACTED***"),
    ],
)
def test_redact_secrets_masks_known_patterns(raw, expected):
    assert redact_secrets(raw) == expected


def test_redact_secrets_masks_inline_within_text():
    text = "authorization: Bearer sk-ant-0987654321xyz for user"
    out = redact_secrets(text)
    assert "sk-ant-0987654321xyz" not in out
    assert "sk-a***REDACTED***" in out
    assert out.startswith("authorization: Bearer ")
    assert out.endswith(" for user")


def test_redact_secrets_passes_through_innocuous_text():
    assert redact_secrets("nothing to see here") == "nothing to see here"


# ---------------------------------------------------------------------------
# exception handling
# ---------------------------------------------------------------------------


def test_json_formatter_includes_exception_info():
    logger = get_logger("exc")
    try:
        raise ValueError("boom")
    except ValueError:
        record = logger.makeRecord(
            name=logger.name,
            level=stdlog.ERROR,
            fn=__file__,
            lno=0,
            msg="explosion",
            args=(),
            exc_info=sys.exc_info(),
        )

    payload = json.loads(JsonFormatter().format(record))
    assert payload["msg"] == "explosion"
    assert "ValueError" in payload["exc_info"]
    assert "boom" in payload["exc_info"]
