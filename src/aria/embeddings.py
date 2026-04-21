"""Pluggable embeddings + semantic memory.

Phase 1 ships a stdlib-only hashing embedder (random projection over token
hashes — fast, deterministic, no deps). Phase 2 swaps in a real embedding
provider (Voyage / local sentence-transformers / OpenAI) behind the same
Embedder protocol.

SemanticMemory stores (text, vector, metadata) rows and serves cosine-
similarity queries. Vectors are stored base64-encoded in SQLite to stay in
one persistence substrate.
"""
from __future__ import annotations

import array
import base64
import hashlib
import math
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]: ...


@dataclass
class HashEmbedder:
    """Feature-hashing embedder over character n-grams.

    Character 3- and 4-grams give us fuzzy lexical similarity: 'deploy' and
    'deployment' share most of their trigrams, so cosine similarity is non-
    trivial without a language model. Not semantic — 'car' and 'automobile'
    still miss — but enough to demonstrate the architecture. Swap the
    Embedder protocol for a real model in Phase 2.
    """
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


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # both are unit-normalized


def _encode(vec: list[float]) -> str:
    return base64.b64encode(array.array("f", vec).tobytes()).decode()


def _decode(s: str) -> list[float]:
    return list(array.array("f", base64.b64decode(s)))


@dataclass
class SemanticMemory:
    db_path: Path
    embedder: Embedder = field(default_factory=HashEmbedder)
    _conn: sqlite3.Connection | None = None

    def __post_init__(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                text TEXT NOT NULL,
                vector TEXT NOT NULL,
                kind TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def remember(self, text: str, kind: str = "note", metadata: dict[str, Any] | None = None) -> int:
        assert self._conn is not None
        vec = self.embedder.embed(text)
        cur = self._conn.execute(
            "INSERT INTO vectors (ts, text, vector, kind, metadata) VALUES (?, ?, ?, ?, ?)",
            (time.time(), text, _encode(vec), kind, _to_json(metadata or {})),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def search(self, query: str, k: int = 5, kind: str | None = None) -> list[dict[str, Any]]:
        assert self._conn is not None
        q_vec = self.embedder.embed(query)
        rows = self._conn.execute(
            "SELECT id, ts, text, vector, kind FROM vectors"
            + (" WHERE kind = ?" if kind else ""),
            (kind,) if kind else (),
        ).fetchall()
        scored: list[tuple[float, dict[str, Any]]] = []
        for rid, ts, text, vec_s, k_kind in rows:
            sim = _cosine(q_vec, _decode(vec_s))
            scored.append((sim, {"id": rid, "ts": ts, "text": text, "kind": k_kind, "score": round(sim, 4)}))
        scored.sort(key=lambda p: p[0], reverse=True)
        return [row for _, row in scored[:k]]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None


def _to_json(obj: Any) -> str:
    import json
    return json.dumps(obj, default=str)
