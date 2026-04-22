"""End-to-end federated CLI tests via subprocess.

Spawns `./mente federated` as a real child process, drives it through stdin,
and asserts on its stdout. These are the only tests that exercise the full
hub + TCP-bus + peer + REPL pipeline in one shot.

Gated off on Windows (federation is macOS / Linux only per ROADMAP.md) and
auto-skipped if the launcher script isn't present in the worktree root.

State safety: each test snapshots any pre-existing `.mente-hub*` /
`.mente-peer*` directories in the project root and restores them afterward,
so a dirty worktree survives the test run intact. Ports are ephemeral so
parallel test workers don't collide on TCP.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import socket
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "mente"

pytestmark = [
    pytest.mark.skipif(os.name == "nt", reason="federated mode is POSIX-only"),
    pytest.mark.skipif(not LAUNCHER.exists(), reason="mente launcher not found in repo root"),
]

SPAWN_TIMEOUT_S = 20.0
STATE_PREFIXES = (".mente-hub", ".mente-peer")


def _find_free_port() -> int:
    """Bind to :0, read the OS-assigned port, release it. Good-enough race-wise."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _state_dirs() -> list[Path]:
    dirs: list[Path] = []
    for prefix in STATE_PREFIXES:
        dirs.extend(REPO_ROOT.glob(f"{prefix}*"))
    return dirs


def _wipe_state() -> None:
    for p in _state_dirs():
        shutil.rmtree(p, ignore_errors=True)


@pytest.fixture
def free_port() -> int:
    """Ephemeral localhost port for the hub's TCP bus."""
    return _find_free_port()


@pytest.fixture
def clean_state(tmp_path: Path) -> Iterator[None]:
    """Snapshot any existing .mente-hub* / .mente-peer* dirs, wipe, restore on exit.

    Keeps the worktree pristine across test runs so an interactive session's
    state isn't clobbered by CI.
    """
    stash = tmp_path / "stash"
    stash.mkdir()
    saved: list[tuple[Path, Path]] = []
    for p in _state_dirs():
        dest = stash / p.name
        shutil.move(str(p), str(dest))
        saved.append((p, dest))
    try:
        yield
    finally:
        _wipe_state()
        for original, dest in saved:
            shutil.move(str(dest), str(original))


async def _run_federated(port: int, stdin_script: str) -> tuple[str, str, int]:
    """Spawn `./mente federated --port <port>`, feed stdin_script, collect output.

    Returns (stdout, stderr, returncode). Always reaps the subprocess, even
    on timeout or test failure.
    """
    proc = await asyncio.create_subprocess_exec(
        str(LAUNCHER),
        "federated",
        "--port",
        str(port),
        cwd=str(REPO_ROOT),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(input=stdin_script.encode()),
            timeout=SPAWN_TIMEOUT_S,
        )
    finally:
        if proc.returncode is None:
            proc.kill()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(proc.wait(), timeout=5.0)
    return (
        stdout_bytes.decode(errors="replace"),
        stderr_bytes.decode(errors="replace"),
        proc.returncode or 0,
    )


def _assert_clean_exit(stdout: str, stderr: str, rc: int) -> None:
    """Shared postconditions: no traceback, zero exit code."""
    combined = stdout + stderr
    detail = f"\nstdout:\n{stdout}\nstderr:\n{stderr}"
    assert "Traceback" not in combined, f"unexpected traceback.{detail}"
    assert rc == 0, f"subprocess exited with {rc}.{detail}"


async def test_federated_boots_and_discovers_peer(free_port: int, clean_state: None) -> None:
    stdout, stderr, rc = await _run_federated(free_port, "/library\n/quit\n")

    assert "discovered: ['peer.math:specialist.math']" in stdout, (
        f"peer discovery line missing.\nstdout:\n{stdout}\nstderr:\n{stderr}"
    )
    _assert_clean_exit(stdout, stderr, rc)


async def test_federated_routes_math_to_remote(free_port: int, clean_state: None) -> None:
    stdout, stderr, rc = await _run_federated(free_port, "what is 7 * 9\n/quit\n")

    # Router attribution appears on stderr (log line); answer text on stdout.
    assert "remote:peer.math:specialist.math" in (stdout + stderr), (
        f"remote reasoner attribution missing.\nstdout:\n{stdout}\nstderr:\n{stderr}"
    )
    assert "7 * 9 = 63" in stdout, f"math answer missing.\nstdout:\n{stdout}\nstderr:\n{stderr}"
    _assert_clean_exit(stdout, stderr, rc)


async def test_federated_reset_between_runs(clean_state: None) -> None:
    stdout_a, stderr_a, rc_a = await _run_federated(_find_free_port(), "/quit\n")
    _assert_clean_exit(stdout_a, stderr_a, rc_a)

    _wipe_state()
    leftovers = _state_dirs()
    assert not leftovers, f"state dirs survived wipe: {leftovers}"

    stdout_b, stderr_b, rc_b = await _run_federated(_find_free_port(), "/quit\n")
    _assert_clean_exit(stdout_b, stderr_b, rc_b)
    assert "discovered: ['peer.math:specialist.math']" in stdout_b, (
        f"second run did not re-discover peer.\nstdout:\n{stdout_b}\nstderr:\n{stderr_b}"
    )
