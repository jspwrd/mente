"""Semantic memory backed by a pluggable Embedder.

The Embedder Protocol and HashEmbedder implementation now live in
`mente.embedders`; they are re-exported here for backward compatibility so
callers can still `from mente.embeddings import HashEmbedder, Embedder`.

SemanticMemory stores (text, vector, metadata) rows and serves cosine-
similarity queries. Vectors are stored base64-encoded in SQLite to stay in
one persistence substrate.
"""
from __future__ import annotations

import array
import base64
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .embedders.hashing import Embedder, HashEmbedder

__all__ = ["Embedder", "HashEmbedder", "SemanticMemory"]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))  # both are unit-normalized


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
        # ``check_same_thread=False`` lets callers offload SemanticMemory
        # calls to asyncio.to_thread / thread pools. mente's event loop is
        # the only writer today, so concurrent-write UB is not a concern;
        # readers are still safe to cross threads.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
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
        rowid = cur.lastrowid
        if rowid is None:  # pragma: no cover - only reachable on WITHOUT ROWID tables
            raise RuntimeError("INSERT produced no rowid; semantic table may be corrupt")
        return rowid

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
