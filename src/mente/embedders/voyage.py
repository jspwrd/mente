"""Voyage AI embedder.

Thin wrapper around `voyageai.Client` that conforms to the Embedder
Protocol. Import of `voyageai` is lazy so installing MENTE stays dep-free
for offline users; the dependency is declared under the optional
`embeddings` extra.
"""
from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from typing import Any

_INSTALL_HINT = (
    "voyageai is not installed. install the embeddings extra: "
    "pip install 'mente[embeddings]' (or pip install voyageai)."
)


class VoyageEmbedder:
    """Voyage API embedder with a small in-process LRU cache."""

    _CACHE_MAX = 1000

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "voyage-3",
        dim: int = 1024,
    ) -> None:
        try:
            import voyageai
        except ImportError as e:
            raise ImportError(_INSTALL_HINT) from e

        self.model = model
        self.dim = dim
        self._api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        # ``voyageai`` doesn't ship type stubs; Client exists but mypy
        # strict can't see it through the dynamic module.
        client_cls: Any = voyageai.Client  # type: ignore[attr-defined]
        self._client = client_cls(api_key=self._api_key) if self._api_key else client_cls()
        self._cache: OrderedDict[str, list[float]] = OrderedDict()

    def _cache_key(self, text: str) -> str:
        return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()

    def _cache_get(self, key: str) -> list[float] | None:
        vec = self._cache.get(key)
        if vec is not None:
            self._cache.move_to_end(key)
        return vec

    def _cache_put(self, key: str, vec: list[float]) -> None:
        self._cache[key] = vec
        self._cache.move_to_end(key)
        if len(self._cache) > self._CACHE_MAX:
            self._cache.popitem(last=False)

    def embed(self, text: str) -> list[float]:
        key = self._cache_key(text)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        result = self._client.embed([text], model=self.model)
        vec = list(result.embeddings[0])
        self._cache_put(key, vec)
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = [[] for _ in texts]
        uncached_idx: list[int] = []
        uncached_texts: list[str] = []
        for i, text in enumerate(texts):
            key = self._cache_key(text)
            cached = self._cache_get(key)
            if cached is not None:
                out[i] = cached
            else:
                uncached_idx.append(i)
                uncached_texts.append(text)
        if uncached_texts:
            result = self._client.embed(uncached_texts, model=self.model)
            for slot, text, emb in zip(
                uncached_idx, uncached_texts, result.embeddings, strict=True
            ):
                vec: list[float] = [float(x) for x in emb]
                out[slot] = vec
                self._cache_put(self._cache_key(text), vec)
        return out
