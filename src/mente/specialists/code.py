"""Code specialist — static critique of a Python snippet.

This is a stdlib-only stub that uses `ast` to surface a handful of common
code smells. It's deliberately conservative: false positives are acceptable
because the router treats specialist output as one signal among many, and a
real code-tuned model will replace this in Phase 2.

Findings surfaced:
  * SyntaxError (parse failure)
  * mutable default arguments (list / dict / set literals)
  * bare `except:` clauses
  * unused imports (imported name never appears as an ast.Name later)
  * undefined names (ast.Name Load that doesn't resolve to any def, assign,
    import, parameter, or builtin in scope)
  * missing return type hints on `def` / `async def`
  * leftover `print(...)` calls
  * TODO / FIXME / XXX comments
"""
from __future__ import annotations

import ast
import builtins
import re
from collections.abc import Iterator
from dataclasses import dataclass

from ..tools import ToolRegistry
from ..types import Intent, ReasonerTier, Response
from ..world_model import WorldModel


_BUILTINS: frozenset[str] = frozenset(dir(builtins))

# Intent-shape detection: these phrases or syntactic markers mean the user
# is probably asking us to look at code.
_TRIGGER_PHRASES = (
    "review this code",
    "review the code",
    "check this",
    "what's wrong with",
    "whats wrong with",
    "what is wrong with",
    "look at this code",
    "debug this",
    "fix this code",
    "critique this",
)

_CODE_TOKEN_RE = re.compile(r"\b(?:def|class|import|from)\s+\w")
_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL)
_TODO_RE = re.compile(r"#.*\b(TODO|FIXME|XXX)\b", re.IGNORECASE)


def _extract_code(text: str) -> str | None:
    """Pull a code block out of free-form intent text.

    Strategy:
      1. Prefer the first triple-backtick fence.
      2. Otherwise, if a trigger phrase appears, grab everything after it.
      3. Otherwise, if the whole text parses as Python, treat it as code.
    """
    fence = _FENCE_RE.search(text)
    if fence:
        return fence.group(1).strip("\n")

    lowered = text.lower()
    for phrase in _TRIGGER_PHRASES:
        idx = lowered.find(phrase)
        if idx != -1:
            tail = text[idx + len(phrase):].lstrip(" :\n\t")
            if tail:
                return tail

    # Last resort: looks code-shaped (has def/class/import) — treat the whole
    # text as code.
    if _CODE_TOKEN_RE.search(text):
        return text

    return None


def _is_code_intent(text: str) -> bool:
    lowered = text.lower()
    if "```" in text:
        return True
    if any(phrase in lowered for phrase in _TRIGGER_PHRASES):
        return True
    if _CODE_TOKEN_RE.search(text):
        return True
    return False


class _ScopeVisitor(ast.NodeVisitor):
    """Walks a module, collecting findings across nested scopes.

    Each scope tracks: parameters, local assignments, and imports visible
    in that scope. We walk the tree once, and when we encounter an
    `ast.Name(Load)` we check it against the chain of enclosing scopes
    plus builtins.
    """

    def __init__(self) -> None:
        self._scopes: list[set[str]] = [set()]
        # module-level imports: name -> line number (for unused-import check)
        self.imports: dict[str, int] = {}
        # names observed as ast.Name(Load) anywhere
        self.loaded_names: set[str] = set()

        self.mutable_defaults: list[tuple[str, int]] = []
        self.bare_excepts: list[int] = []
        self.undefined_names: list[tuple[str, int]] = []
        self.missing_return_hints: list[tuple[str, int]] = []
        self.print_calls: list[int] = []

    # ---- scope helpers ---------------------------------------------------
    def _is_defined(self, name: str) -> bool:
        return any(name in s for s in self._scopes)

    def _define(self, name: str) -> None:
        self._scopes[-1].add(name)

    def _push(self) -> None:
        self._scopes.append(set())

    def _pop(self) -> None:
        self._scopes.pop()

    def _preseed(self, body: list[ast.stmt]) -> None:
        """Define names introduced by top-level statements of a body so that
        forward references (e.g. mutual recursion, methods calling each other)
        resolve correctly."""
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self._define(stmt.name)
            elif isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    for nm in _collect_assign_names(tgt):
                        self._define(nm)
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                self._define(stmt.target.id)

    def prime_module(self, module: ast.Module) -> None:
        self._preseed(module.body)
        for node in module.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    if isinstance(node, ast.Import):
                        bound = alias.asname or alias.name.split(".")[0]
                    else:
                        bound = alias.asname or alias.name
                    self._define(bound)
                    self.imports[bound] = node.lineno

    # ---- visitors --------------------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            bound = alias.asname or alias.name.split(".")[0]
            self._define(bound)
            # Only record top-level for unused-import (nested imports are
            # intentional and rarely unused in practice).
            if len(self._scopes) == 1:
                self.imports.setdefault(bound, node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "*":
                continue
            bound = alias.asname or alias.name
            self._define(bound)
            if len(self._scopes) == 1:
                self.imports.setdefault(bound, node.lineno)
        self.generic_visit(node)

    def _visit_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        # function name visible in enclosing scope
        self._define(node.name)

        # mutable-default check (in enclosing scope — defaults evaluate there)
        for default in list(node.args.defaults) + list(node.args.kw_defaults):
            if default is None:
                continue
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                self.mutable_defaults.append((node.name, default.lineno))
            # Defaults evaluate in the enclosing scope; visit so their names
            # count as loads.
            self.visit(default)

        # visit decorators in enclosing scope as well
        for dec in node.decorator_list:
            self.visit(dec)

        # annotations are loads in the enclosing scope
        for arg in _iter_args(node.args):
            if arg.annotation is not None:
                self.visit(arg.annotation)
        if node.returns is not None:
            self.visit(node.returns)

        # missing return type hint
        if node.returns is None and node.name != "__init__":
            self.missing_return_hints.append((node.name, node.lineno))

        self._push()
        for arg in _iter_args(node.args):
            self._define(arg.arg)
        self._preseed(node.body)
        for stmt in node.body:
            self.visit(stmt)
        self._pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._define(node.name)
        self._push()
        self._preseed(node.body)
        for stmt in node.body:
            self.visit(stmt)
        self._pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        # visit value first (so RHS names are loads, not yet defined)
        self.visit(node.value)
        for tgt in node.targets:
            for nm in _collect_assign_names(tgt):
                self._define(nm)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
        if node.annotation is not None:
            self.visit(node.annotation)
        if isinstance(node.target, ast.Name):
            self._define(node.target.id)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            # AugAssign both loads and stores — we count it as a load *and* define.
            self.loaded_names.add(node.target.id)
            self._define(node.target.id)

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        for nm in _collect_assign_names(node.target):
            self._define(nm)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def _visit_comp(self, node: ast.AST) -> None:
        # ListComp/SetComp/DictComp/GeneratorExp introduce a new scope in
        # Python 3 (except the first iterable). Approximate it.
        gens: list[ast.comprehension] = getattr(node, "generators", [])
        self._push()
        for gen in gens:
            self.visit(gen.iter)
            for nm in _collect_assign_names(gen.target):
                self._define(nm)
            for if_clause in gen.ifs:
                self.visit(if_clause)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            elt = getattr(node, "elt", None)
            if elt is not None:
                self.visit(elt)
        self._pop()

    def visit_ListComp(self, node: ast.ListComp) -> None: self._visit_comp(node)
    def visit_SetComp(self, node: ast.SetComp) -> None: self._visit_comp(node)
    def visit_DictComp(self, node: ast.DictComp) -> None: self._visit_comp(node)
    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None: self._visit_comp(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._push()
        for arg in _iter_args(node.args):
            self._define(arg.arg)
        self.visit(node.body)
        self._pop()

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self.bare_excepts.append(node.lineno)
        else:
            self.visit(node.type)
        if node.name is not None:
            self._define(node.name)
        for stmt in node.body:
            self.visit(stmt)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                for nm in _collect_assign_names(item.optional_vars):
                    self._define(nm)
        for stmt in node.body:
            self.visit(stmt)

    def visit_Global(self, node: ast.Global) -> None:
        for nm in node.names:
            self._define(nm)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        for nm in node.names:
            self._define(nm)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            self.print_calls.append(node.lineno)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.loaded_names.add(node.id)
            if node.id not in _BUILTINS and not self._is_defined(node.id):
                self.undefined_names.append((node.id, node.lineno))


def _iter_args(args: ast.arguments) -> Iterator[ast.arg]:
    yield from args.posonlyargs
    yield from args.args
    if args.vararg is not None:
        yield args.vararg
    yield from args.kwonlyargs
    if args.kwarg is not None:
        yield args.kwarg


def _collect_assign_names(target: ast.AST) -> list[str]:
    """Walk an assignment target and yield every bound name."""
    names: list[str] = []
    if isinstance(target, ast.Name):
        names.append(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            names.extend(_collect_assign_names(elt))
    elif isinstance(target, ast.Starred):
        names.extend(_collect_assign_names(target.value))
    return names


def _scan_todos(code: str) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    for i, line in enumerate(code.splitlines(), start=1):
        m = _TODO_RE.search(line)
        if m:
            found.append((m.group(1).upper(), i))
    return found


def _analyze(code: str) -> tuple[list[str], str | None]:
    """Return (findings, syntax_error_message). Only one of the two is populated."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        msg = f"SyntaxError at line {e.lineno}: {e.msg}"
        return [], msg

    visitor = _ScopeVisitor()
    visitor.prime_module(tree)
    for stmt in tree.body:
        visitor.visit(stmt)

    findings: list[str] = []

    for fn_name, line in visitor.mutable_defaults:
        findings.append(
            f"mutable default argument in `{fn_name}` (line {line}); "
            "use `None` and initialize inside the body."
        )
    for line in visitor.bare_excepts:
        findings.append(
            f"bare `except:` at line {line}; catch a specific exception type."
        )
    for name, line in sorted(visitor.imports.items(), key=lambda kv: kv[1]):
        if name not in visitor.loaded_names:
            findings.append(f"unused import `{name}` at line {line}.")
    # dedupe undefined names (same name can appear many times)
    seen: set[str] = set()
    for name, line in visitor.undefined_names:
        if name in seen:
            continue
        seen.add(name)
        findings.append(f"undefined name `{name}` at line {line}.")
    for fn_name, line in visitor.missing_return_hints:
        findings.append(
            f"function `{fn_name}` (line {line}) is missing a return type hint."
        )
    if visitor.print_calls:
        lines = ", ".join(str(ln) for ln in visitor.print_calls)
        findings.append(
            f"`print(...)` call(s) at line(s) {lines}; likely debug leftovers."
        )
    for tag, line in _scan_todos(code):
        findings.append(f"{tag} comment at line {line}.")

    return findings, None


@dataclass
class CodeSpecialist:
    name: str = "specialist.code"
    tier: ReasonerTier = "specialist"
    est_cost_ms: float = 25.0

    async def answer(
        self, intent: Intent, world: WorldModel, tools: ToolRegistry
    ) -> Response:
        if not _is_code_intent(intent.text):
            return Response(
                text="", reasoner=self.name, tier=self.tier,
                confidence=0.0, cost_ms=self.est_cost_ms,
            )

        code = _extract_code(intent.text)
        if not code or not code.strip():
            return Response(
                text="", reasoner=self.name, tier=self.tier,
                confidence=0.0, cost_ms=self.est_cost_ms,
            )

        findings, syntax_err = _analyze(code)
        if syntax_err is not None:
            return Response(
                text=(
                    "I reviewed the code and found a parse error: "
                    f"{syntax_err}. Fix the syntax before deeper review is useful."
                ),
                reasoner=self.name, tier=self.tier,
                confidence=0.95, cost_ms=self.est_cost_ms,
            )

        if not findings:
            return Response(
                text="I reviewed the code and found no obvious issues.",
                reasoner=self.name, tier=self.tier,
                confidence=0.9, cost_ms=self.est_cost_ms,
            )

        bullets = "\n".join(f"  - {f}" for f in findings)
        return Response(
            text="I reviewed the code and found:\n" + bullets,
            reasoner=self.name, tier=self.tier,
            confidence=0.95, cost_ms=self.est_cost_ms,
        )
