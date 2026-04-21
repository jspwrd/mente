"""Typed tool registry.

Tools are first-class operators. Each has a typed signature and a cost/latency
estimate used by the router.

Phase 1: tools are plain async Python callables invoked explicitly by reasoners.
Phase 2: tools become a latent modality — their invocations are emitted as
structured attention outputs and their returns re-enter the decoding loop as
embeddings, not strings.
"""
from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolSpec:
    name: str
    description: str
    params: dict[str, str]  # param_name -> type_hint string
    returns: str
    est_cost_ms: float
    fn: Callable[..., Awaitable[Any]]


@dataclass
class ToolResult:
    tool: str
    ok: bool
    value: Any
    error: str | None
    cost_ms: float


@dataclass
class ToolRegistry:
    _tools: dict[str, ToolSpec] = field(default_factory=dict)

    def register(
        self,
        name: str,
        description: str,
        returns: str,
        est_cost_ms: float = 5.0,
    ) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
        def deco(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
            sig = inspect.signature(fn)
            params = {
                pname: str(p.annotation) if p.annotation is not inspect.Parameter.empty else "Any"
                for pname, p in sig.parameters.items()
            }
            self._tools[name] = ToolSpec(
                name=name,
                description=description,
                params=params,
                returns=returns,
                est_cost_ms=est_cost_ms,
                fn=fn,
            )
            return fn
        return deco

    def list(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    async def invoke(self, name: str, **kwargs: Any) -> ToolResult:
        spec = self._tools.get(name)
        if spec is None:
            return ToolResult(tool=name, ok=False, value=None, error=f"unknown tool: {name}", cost_ms=0.0)
        t0 = time.perf_counter()
        try:
            value = await spec.fn(**kwargs)
            return ToolResult(tool=name, ok=True, value=value, error=None, cost_ms=(time.perf_counter() - t0) * 1000)
        except Exception as e:
            return ToolResult(tool=name, ok=False, value=None, error=repr(e), cost_ms=(time.perf_counter() - t0) * 1000)
