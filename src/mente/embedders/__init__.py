"""Pluggable embedder implementations.

The `Embedder` Protocol is the interface. Three shipping impls:

- `HashEmbedder` — offline, stdlib-only char n-gram hashing; default.
- `VoyageEmbedder` — real semantic embeddings via Voyage API; behind
  the `embeddings` extra.
- `LocalEmbedder` — CPU-friendly sentence-transformers; behind the
  `embeddings-local` extra. Best choice for offline users who want
  real semantic similarity without an API key.

Both `VoyageEmbedder` and `LocalEmbedder` are lazily imported so just
importing this package doesn't require their (heavy) dependencies.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .hashing import Embedder, HashEmbedder

if TYPE_CHECKING:  # pragma: no cover
    from .local import LocalEmbedder
    from .voyage import VoyageEmbedder

__all__ = ["Embedder", "HashEmbedder", "LocalEmbedder", "VoyageEmbedder"]


def __getattr__(name: str) -> Any:
    if name == "VoyageEmbedder":
        from .voyage import VoyageEmbedder
        return VoyageEmbedder
    if name == "LocalEmbedder":
        from .local import LocalEmbedder
        return LocalEmbedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
