"""Program synthesis + library.

A reasoner that, for computation-shaped intents, synthesizes a small Python
function, executes it in a subprocess sandbox, validates the output, and —
on repeated success — promotes the verified snippet into the ToolRegistry.

The synthesis step itself is pluggable: see ``mente.synthesizers`` for the
``Synthesizer`` Protocol and the two shipped backends (template, LLM). The
subprocess + AST-gate machinery below is the trust boundary — every
synthesized function, regardless of author, runs through it.
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import inspect
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .resilience import retry_async
from .synthesizers import Synthesizer, TemplateSynthesizer
from .tools import ToolRegistry
from .types import Intent, ReasonerTier, Response
from .world_model import WorldModel


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


async def _run_sandboxed(source: str, entrypoint: str, args: dict,
                         timeout_s: float = 2.0) -> dict:
    """Run `source` in a subprocess with -S (no site), returning {ok, value|error}."""
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
    proc = await retry_async(
        attempts=2, retry_on=(OSError, ChildProcessError),
    )(asyncio.create_subprocess_exec)(
        sys.executable, "-S", "-I", "-c", driver,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(
            proc.communicate(json.dumps(args).encode()), timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return {"ok": False, "error": f"timeout after {timeout_s}s"}

    if proc.returncode != 0:
        return {"ok": False, "error": err.decode().strip() or "nonzero exit"}
    try:
        return json.loads(out.decode().strip().splitlines()[-1])
    except Exception as e:
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
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self._primitives = {k: Primitive(**v) for k, v in data.items()}

    def save(self) -> None:
        self.path.write_text(
            json.dumps({k: v.__dict__ for k, v in self._primitives.items()}, indent=2)
        )

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
        except Exception as e:
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
        async def _invoke(**kwargs):
            result = await _run_sandboxed(prim.source, prim.entrypoint, kwargs)
            if not result.get("ok"):
                raise RuntimeError(result.get("error"))
            return result["value"]

        # The tool signature mirrors the primitive's args so the registry
        # records it like any other tool.
        self.tools._tools[prim.name] = self.tools._tools.get(prim.name) or _make_spec(prim, _invoke)


def _make_spec(prim: Primitive, fn) -> "object":
    from .tools import ToolSpec
    return ToolSpec(
        name=prim.name,
        description=f"Synthesized primitive: {prim.entrypoint}",
        params=prim.signature,
        returns="Any",
        est_cost_ms=30.0,
        fn=fn,
    )
