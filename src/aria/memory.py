"""Two-tier memory: fast (in-proc, TTL) and slow (SQLite, append-only episodic).

Fast memory = working set, decays quickly. Slow memory = the day's log of events.
Semantic memory (vectors) is a Phase 2 expansion — stubbed here with a
text-similarity fallback so retrieval works without embeddings.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FastMemory:
    """In-process scratchpad with optional TTL per key."""
    _data: dict[str, tuple[Any, float | None, float]] = field(default_factory=dict)

    def set(self, key: str, value: Any, ttl_s: float | None = None) -> None:
        self._data[key] = (value, ttl_s, time.time())

    def get(self, key: str) -> Any:
        if key not in self._data:
            return None
        value, ttl, ts = self._data[key]
        if ttl is not None and time.time() - ts >= ttl:
            del self._data[key]
            return None
        return value

    def all(self) -> dict[str, Any]:
        now = time.time()
        return {
            k: v for k, (v, ttl, ts) in self._data.items()
            if ttl is None or now - ts < ttl
        }


@dataclass
class SlowMemory:
    """Append-only episodic log backed by SQLite."""
    db_path: Path
    _conn: sqlite3.Connection | None = None

    def __post_init__(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                kind TEXT NOT NULL,
                actor TEXT NOT NULL,
                payload TEXT NOT NULL,
                trace_id TEXT
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON episodes(ts)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_kind ON episodes(kind)")
        self._conn.commit()

    def record(self, kind: str, actor: str, payload: dict[str, Any], trace_id: str = "") -> None:
        assert self._conn is not None
        self._conn.execute(
            "INSERT INTO episodes (ts, kind, actor, payload, trace_id) VALUES (?, ?, ?, ?, ?)",
            (time.time(), kind, actor, json.dumps(payload, default=str), trace_id),
        )
        self._conn.commit()

    def query(self, kind: str | None = None, since: float | None = None, limit: int = 100) -> list[dict[str, Any]]:
        assert self._conn is not None
        q = "SELECT ts, kind, actor, payload, trace_id FROM episodes WHERE 1=1"
        args: list[Any] = []
        if kind:
            q += " AND kind = ?"
            args.append(kind)
        if since:
            q += " AND ts >= ?"
            args.append(since)
        q += " ORDER BY ts DESC LIMIT ?"
        args.append(limit)
        rows = self._conn.execute(q, args).fetchall()
        return [
            {"ts": r[0], "kind": r[1], "actor": r[2], "payload": json.loads(r[3]), "trace_id": r[4]}
            for r in rows
        ]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
