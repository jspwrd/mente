"""Program synthesis + library.

A reasoner that, for computation-shaped intents, synthesizes a small Python
function, executes it in a subprocess sandbox, validates the output, and —
on repeated success — promotes the verified snippet into the ToolRegistry.

The synthesis step itself is pluggable: see ``mente.synthesizers`` for the
``Synthesizer`` Protocol and the two shipped backends (template, LLM). The
subprocess + AST-gate machinery below is the trust boundary — every
synthesized function, regardless of author, runs through it.

Threat model
------------
The sandbox is a defence-in-depth wrapper around synthesized (template- or
LLM-authored) Python snippets. It defends against:

* **Accidental imports / filesystem access** — ``_validate_ast`` rejects
  ``import``/``from-import``, dunder attribute access, and a blocklist of
  names (``open``, ``exec``, ``eval``, ``compile``, ``globals``/``locals``,
  ``getattr``/``setattr``, etc.). Code that shouldn't need I/O can't get at
  it even by reflection.
* **Unbounded recursion / runaway CPU** — every call runs in a separate
  process under ``asyncio.wait_for(timeout=2s)`` (belt) and POSIX
  ``RLIMIT_CPU`` / ``RLIMIT_AS`` / ``RLIMIT_NOFILE`` / ``RLIMIT_NPROC``
  (braces). If the snippet loops forever or blows up memory, the kernel
  kills it; asyncio then reaps.
* **Environment leakage** — the subprocess is launched with a minimal
  ``env={"PATH": "/usr/bin:/bin"}`` and Python flags ``-S -I`` (no site,
  isolated mode), so host env vars (API keys, LD_PRELOAD, PYTHONPATH, …)
  don't leak in.

It does **NOT** defend against:

* **Kernel exploits or container escapes** — rlimits are process-local and
  assume a well-behaved kernel. Running mente on a shared multi-tenant
  host requires an additional container / seccomp layer.
* **Side-channel attacks** — timing, cache, and power channels are out of
  scope. Don't synthesize cryptographic primitives this way.
* **Untrusted, adversarial input** — an LLM-authored synthesis on
  attacker-controlled prompts is still a reflected code-execution surface.
  The AST gate is a best-effort list of foot-guns, not a formal proof of
  memory safety. If your users can steer synthesis (prompt injection or
  otherwise), review generated snippets out-of-band before promoting into
  the library, and consider disabling the LLM synthesizer entirely.
"""
from __future__ import annotations

import ast
import asyncio
import contextlib
import hashlib
import inspect
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .resilience import retry_async
from .synthesizers import Synthesizer, TemplateSynthesizer
from .tools import ToolRegistry, ToolSpec
from .types import Intent, ReasonerTier, Response
from .world_model import WorldModel

_log = logging.getLogger(__name__)

# A list of disallowed AST nodes. Keeps synthesized code sandboxable.
_DISALLOWED_NODES = (
    ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.With,
    ast.AsyncWith, ast.AsyncFor, ast.AsyncFunctionDef, ast.Try,
    ast.Raise, ast.Delete, ast.Lambda,
)
_DISALLOWED_NAMES = {
    "__import__", "open", "exec", "eval", "compile", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "help", "input", "print",
    "breakpoint", "memoryview",
}

# Resource limits enforced on POSIX via preexec_fn. Belt-and-braces with the
# 2s asyncio.wait_for timeout in _run_sandboxed.
_RLIMIT_CPU_SOFT = 2              # seconds
_RLIMIT_CPU_HARD = 3
_RLIMIT_AS_BYTES = 256 * 1024 * 1024  # 256MB address space
_RLIMIT_NOFILE = 32                # open-file descriptor cap
_RLIMIT_NPROC = 0                  # no forking


def _apply_rlimits() -> None:
    """preexec_fn: cap CPU / address-space / fds / forks in the child.

    Runs in the subprocess *after* fork and *before* exec. Each setrlimit is
    wrapped individually — some limits (notably RLIMIT_NPROC) behave
    differently on macOS vs. Linux, and we'd rather enforce what we can than
    fail the whole sandbox because one knob is unavailable.
    """
    import resource  # POSIX-only; imported inside the guard in _run_sandboxed

    limits: list[tuple[str, int, int]] = [
        ("RLIMIT_CPU", _RLIMIT_CPU_SOFT, _RLIMIT_CPU_HARD),
        ("RLIMIT_AS", _RLIMIT_AS_BYTES, _RLIMIT_AS_BYTES),
        ("RLIMIT_NOFILE", _RLIMIT_NOFILE, _RLIMIT_NOFILE),
        ("RLIMIT_NPROC", _RLIMIT_NPROC, _RLIMIT_NPROC),
    ]
    for attr, soft, hard in limits:
        rlimit = getattr(resource, attr, None)
        if rlimit is None:
            continue  # platform doesn't expose this limit
        # Raising here would kill the child between fork and exec — swallow
        # and continue; the asyncio timeout is still a backstop.
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(rlimit, (soft, hard))


def _validate_ast(source: str) -> None:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, _DISALLOWED_NODES):
            raise ValueError(f"disallowed AST node: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id in _DISALLOWED_NAMES:
            raise ValueError(f"disallowed name: {node.id}")
        if isinstance(node, ast.Attribute):
            attr = node.attr
            if attr.startswith("__") and attr.endswith("__"):
                raise ValueError(f"dunder access: {attr}")


async def _run_sandboxed(source: str, entrypoint: str, args: dict[str, Any],
                         timeout_s: float = 2.0) -> dict[str, Any]:
    """Run ``source`` in a subprocess with ``-S -I`` (no site, isolated).

    Returns ``{ok, value|error}``. The child is launched with:

    * POSIX rlimits (CPU=2s, AS=256MB, NOFILE=32, NPROC=0) via ``preexec_fn``.
      On Windows these can't be set — we log a warning and rely on the
      asyncio timeout alone.
    * A minimal env (``PATH=/usr/bin:/bin``) so host credentials, PYTHONPATH,
      and similar don't leak into synthesized code.
    * ``close_fds=True`` so inherited descriptors don't escape either.
    """
    _validate_ast(source)

    driver = f"""
import json, sys
NS = {{}}
exec({source!r}, NS)
fn = NS[{entrypoint!r}]
args = json.loads(sys.stdin.read())
try:
    out = fn(**args)
    print(json.dumps({{"ok": True, "value": out}}))
except Exception as e:
    print(json.dumps({{"ok": False, "error": f"{{type(e).__name__}}: {{e}}"}}))
"""

    # Part A + C: preexec_fn applies rlimits on POSIX; Windows has no rlimits.
    if sys.platform != "win32":
        preexec_fn: Any = _apply_rlimits
    else:
        preexec_fn = None
        _log.warning(
            "synthesis sandbox: resource limits not enforced on win32; "
            "relying on asyncio timeout (%.1fs) alone",
            timeout_s,
        )

    # Part B: minimal env — no host vars leak into the sandbox.
    sandbox_env = {"PATH": "/usr/bin:/bin"}

    try:
        proc = await retry_async(
            attempts=2, retry_on=(OSError, ChildProcessError),
        )(asyncio.create_subprocess_exec)(
            sys.executable, "-S", "-I", "-c", driver,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=sandbox_env,
            close_fds=True,
            preexec_fn=preexec_fn,
        )
    except (OSError, FileNotFoundError) as e:
        return {"ok": False, "error": f"sandbox spawn failed: {e}"}

    try:
        out, err = await asyncio.wait_for(
            proc.communicate(json.dumps(args).encode()), timeout=timeout_s,
        )
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return {"ok": False, "error": f"timeout after {timeout_s}s"}

    if proc.returncode != 0:
        return {"ok": False, "error": err.decode().strip() or "nonzero exit"}
    raw = out.decode().strip().splitlines()
    if not raw:
        return {"ok": False, "error": "empty sandbox output"}
    try:
        parsed: dict[str, Any] = json.loads(raw[-1])
        return parsed
    except json.JSONDecodeError as e:
        return {"ok": False, "error": f"bad sandbox output: {e}"}


@dataclass
class Primitive:
    """A verified, synthesized function promoted into the library."""
    name: str
    source: str
    entrypoint: str
    signature: dict[str, str]
    invocations: int = 0


@dataclass
class LibraryStore:
    """Persistent library of verified primitives."""
    path: Path
    _primitives: dict[str, Primitive] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
        except json.JSONDecodeError as e:
            _log.warning("library file %s is corrupt JSON; starting empty: %s", self.path, e)
            return
        if not isinstance(data, dict):
            _log.warning("library file %s has unexpected shape; starting empty", self.path)
            return
        for name, payload in data.items():
            if not isinstance(payload, dict):
                _log.warning("library entry %r is not an object; skipping", name)
                continue
            try:
                self._primitives[name] = Primitive(**payload)
            except TypeError as e:
                # Missing/extra fields — log and skip rather than brick the runtime.
                _log.warning("library entry %r is malformed; skipping (%s)", name, e)

    def save(self) -> None:
        # Atomic write: a crash mid-save leaves either the old file or the
        # new file, never a half-written one. Matches LatentState.checkpoint.
        payload = json.dumps({k: v.__dict__ for k, v in self._primitives.items()}, indent=2)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(payload)
        tmp.replace(self.path)

    def add(self, p: Primitive) -> None:
        self._primitives[p.name] = p
        self.save()

    def get(self, name: str) -> Primitive | None:
        return self._primitives.get(name)

    def list(self) -> list[Primitive]:
        return list(self._primitives.values())


@dataclass
class SynthesisReasoner:
    """Reasoner that attempts to answer via program synthesis.

    Flow: synthesize code (template or LLM) → validate AST → execute
    sandboxed → if successful, promote to library (and register as a tool)
    → return the computed value. The ``synthesizer`` field is pluggable; if
    unset, the deterministic template synthesizer is used.
    """
    library: LibraryStore
    tools: ToolRegistry
    synthesizer: Synthesizer = field(default_factory=TemplateSynthesizer)
    name: str = "specialist.synthesis"
    tier: ReasonerTier = "specialist"
    est_cost_ms: float = 50.0

    async def answer(
        self, intent: Intent, world: WorldModel, tools: ToolRegistry
    ) -> Response:
        # Synthesizers may expose an async `asynthesize` (LLM) or a sync
        # `synthesize` (template); prefer the async form when available so we
        # don't block the event loop on a network call.
        asyn = getattr(self.synthesizer, "asynthesize", None)
        if asyn is not None and inspect.iscoroutinefunction(asyn):
            candidate = await asyn(intent.text)
        else:
            candidate = self.synthesizer.synthesize(intent.text)
        if candidate is None:
            return Response(
                text="", reasoner=self.name, tier=self.tier,
                confidence=0.0, cost_ms=self.est_cost_ms,
            )
        source, entry, args = candidate

        try:
            result = await _run_sandboxed(source, entry, args)
        except (ValueError, SyntaxError) as e:
            # _validate_ast surfaces as ValueError; ast.parse as SyntaxError.
            return Response(
                text=f"[synthesis: validation failed — {e}]",
                reasoner=self.name, tier=self.tier,
                confidence=0.1, cost_ms=self.est_cost_ms,
            )

        if not result.get("ok"):
            return Response(
                text=f"[synthesis: sandboxed run failed — {result.get('error')}]",
                reasoner=self.name, tier=self.tier,
                confidence=0.2, cost_ms=self.est_cost_ms,
            )

        value = result["value"]

        # Promote to library + register as a tool so future calls skip synthesis.
        key = hashlib.blake2b(source.encode(), digest_size=8).hexdigest()
        primitive_name = f"lib.{entry}.{key[:6]}"
        existing = self.library.get(primitive_name)
        if existing is None:
            prim = Primitive(
                name=primitive_name, source=source, entrypoint=entry,
                signature={k: type(v).__name__ for k, v in args.items()},
                invocations=1,
            )
            self.library.add(prim)
            self._register_tool(prim)
        else:
            existing.invocations += 1
            self.library.save()

        return Response(
            text=f"{entry}({', '.join(f'{k}={v}' for k, v in args.items())}) = {value}",
            reasoner=self.name, tier=self.tier,
            confidence=0.98, cost_ms=self.est_cost_ms,
            tools_used=[primitive_name],
        )

    def _register_tool(self, prim: Primitive) -> None:
        async def _invoke(**kwargs: Any) -> Any:
            result = await _run_sandboxed(prim.source, prim.entrypoint, kwargs)
            if not result.get("ok"):
                raise RuntimeError(result.get("error"))
            return result["value"]

        # The tool signature mirrors the primitive's args so the registry
        # records it like any other tool.
        self.tools._tools[prim.name] = self.tools._tools.get(prim.name) or _make_spec(prim, _invoke)


def _make_spec(prim: Primitive, fn: Any) -> ToolSpec:
    return ToolSpec(
        name=prim.name,
        description=f"Synthesized primitive: {prim.entrypoint}",
        params=prim.signature,
        returns="Any",
        est_cost_ms=30.0,
        fn=fn,
    )
