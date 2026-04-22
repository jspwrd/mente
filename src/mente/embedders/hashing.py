"""Hashing-based embedder.

Stdlib-only fallback. Character 3- and 4-grams give us fuzzy lexical
similarity: 'deploy' and 'deployment' share most of their trigrams, so
cosine similarity is non-trivial without a language model. Not semantic —
'car' and 'automobile' still miss — but enough to run fully offline.

This module also defines the ``Embedder`` Protocol that every embedder
backend in ``mente.embedders`` satisfies (it is re-exported from
``mente.embedders`` for convenience).

Drop-in implementation example::

    from dataclasses import dataclass

    @dataclass
    class ConstantEmbedder:
        dim: int = 16

        def embed(self, text: str) -> list[float]:
            v = [1.0] + [0.0] * (self.dim - 1)
            return v  # already unit-norm
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Protocol


class Embedder(Protocol):
    """Pluggable text-to-vector embedder.

    An Embedder maps a string to a fixed-length vector of floats. mente
    calls it from the semantic-memory path (``memory.search``, consolidator
    recall, curiosity ranking) and from any place that needs cosine
    similarity between text fragments.

    Invariants:
        * Every returned vector MUST have length ``self.dim``.
        * Vectors MUST be unit-normalized (L2 norm ``== 1.0``) so callers
          can treat dot product as cosine similarity directly. The sole
          exception is the zero vector, which is allowed for empty/invalid
          inputs (callers handle the degenerate case).
        * ``embed`` is synchronous and MUST be cheap enough to call inline;
          async I/O-bound backends should batch internally or expose a
          separate async façade.

    Attributes:
        dim: Dimensionality of the produced vectors. Must match the shape
            expected by any vector store the embedder is paired with.
    """

    dim: int

    def embed(self, text: str) -> list[float]:
        """Embed ``text`` into a unit-norm vector of length ``self.dim``.

        Args:
            text: The input string. May be empty; implementations should
                return a zero vector (length ``self.dim``) rather than
                raising.

        Returns:
            A ``list[float]`` of length ``self.dim``. Unit-normalized
            unless ``text`` produced no features, in which case a
            zero-vector of the same length is returned.

        Raises:
            Exception: Only for backend failures (e.g. remote API errors
                in network-backed embedders). Stdlib backends should not
                raise on normal inputs.
        """
        ...


@dataclass
class HashEmbedder:
    """Feature-hashing embedder over character n-grams."""
    dim: int = 256
    ngram_sizes: tuple[int, ...] = (3, 4)

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            return vec
        for tok in tokens:
            padded = f" {tok} "
            for n in self.ngram_sizes:
                if len(padded) < n:
                    continue
                for i in range(len(padded) - n + 1):
                    gram = padded[i:i + n]
                    h = hashlib.blake2b(gram.encode(), digest_size=8).digest()
                    idx = int.from_bytes(h[:4], "little") % self.dim
                    sign = 1.0 if h[4] & 1 else -1.0
                    vec[idx] += sign
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]
