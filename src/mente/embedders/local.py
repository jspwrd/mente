"""Local sentence-transformers embedder.

A CPU-friendly option for users who want real semantic similarity without an
API key. Requires the ``embeddings-local`` extra:

    pip install 'mente[embeddings-local]'
    uv add 'mente[embeddings-local]'

The default model (``all-MiniLM-L6-v2``, 384 dims) is tiny (~90MB) and fast
enough for interactive use on a laptop CPU. Swap the ``model`` constructor
arg for any other sentence-transformers checkpoint if you need higher quality.
"""
from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from typing import Any

_INSTALL_HINT = (
    "sentence-transformers is not installed. install the embeddings-local extra: "
    "pip install 'mente[embeddings-local]' (or pip install sentence-transformers)."
)


class LocalEmbedder:
    """sentence-transformers embedder with a small LRU cache.

    Output vectors are L2-normalized so the existing cosine-similarity code
    path (``sum(x*y ...)``) works unchanged.
    """

    _CACHE_MAX = 1000

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        dim: int | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise ImportError(_INSTALL_HINT) from e

        self.model_name = model
        self._model: Any = SentenceTransformer(model)
        # Allow override; otherwise ask the model.
        self.dim = dim if dim is not None else int(self._model.get_sentence_embedding_dimension())
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

    @staticmethod
    def _normalize(vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed(self, text: str) -> list[float]:
        key = self._cache_key(text)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        arr = self._model.encode(text, convert_to_numpy=True)
        vec = self._normalize([float(x) for x in arr])
        self._cache_put(key, vec)
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = [[] for _ in texts]
        uncached_idx: list[int] = []
        uncached_texts: list[str] = []
        for i, text in enumerate(texts):
            cached = self._cache_get(self._cache_key(text))
            if cached is not None:
                out[i] = cached
            else:
                uncached_idx.append(i)
                uncached_texts.append(text)
        if uncached_texts:
            arrs = self._model.encode(uncached_texts, convert_to_numpy=True)
            for slot, text, arr in zip(uncached_idx, uncached_texts, arrs, strict=True):
                vec = self._normalize([float(x) for x in arr])
                out[slot] = vec
                self._cache_put(self._cache_key(text), vec)
        return out
