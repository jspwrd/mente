"""Hashing-based embedder.

Stdlib-only fallback. Character 3- and 4-grams give us fuzzy lexical
similarity: 'deploy' and 'deployment' share most of their trigrams, so
cosine similarity is non-trivial without a language model. Not semantic —
'car' and 'automobile' still miss — but enough to run fully offline.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Protocol


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]: ...


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
