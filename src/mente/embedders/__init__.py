"""Pluggable embedder implementations.

The `Embedder` Protocol is the interface; `HashEmbedder` is the offline
default; `VoyageEmbedder` is the real-model option (gated behind the
`embeddings` extra).

`VoyageEmbedder` is lazily imported so that just importing this package
does not require `voyageai` to be installed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .hashing import Embedder, HashEmbedder

if TYPE_CHECKING:  # pragma: no cover
    from .voyage import VoyageEmbedder

__all__ = ["Embedder", "HashEmbedder", "VoyageEmbedder"]


def __getattr__(name: str) -> Any:
    if name == "VoyageEmbedder":
        from .voyage import VoyageEmbedder
        return VoyageEmbedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
