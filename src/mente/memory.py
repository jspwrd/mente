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
    """In-process scratchpad with optional TTL per key.

    A working-set cache — process-local, no persistence. Each entry is stored
    with its insertion timestamp so that reads past the TTL lazily evict.
    Used by the runtime for short-lived intermediate state that should not
    survive a restart.
    """
    _data: dict[str, tuple[Any, float | None, float]] = field(default_factory=dict)

    def set(self, key: str, value: Any, ttl_s: float | None = None) -> None:
        """Store ``value`` under ``key``, optionally expiring after ``ttl_s``.

        Args:
            key: Lookup identifier. Overwrites any existing entry.
            value: Arbitrary Python object to cache.
            ttl_s: Seconds until the entry is considered expired. ``None``
                means the entry lives until the process exits.
        """
        self._data[key] = (value, ttl_s, time.time())

    def get(self, key: str) -> Any:
        """Return the value for ``key``, or ``None`` if missing or expired.

        Expired entries are evicted as a side effect so subsequent calls do
        not re-check the TTL.

        Args:
            key: Lookup identifier.

        Returns:
            The cached value, or ``None`` if the key is absent or its TTL
            has elapsed.
        """
        if key not in self._data:
            return None
        value, ttl, ts = self._data[key]
        if ttl is not None and time.time() - ts >= ttl:
            del self._data[key]
            return None
        return value

    def all(self) -> dict[str, Any]:
        """Return a snapshot dict of all live (non-expired) entries.

        The returned dict is a fresh copy; callers may mutate it freely.
        Expired entries are filtered out but not evicted from the backing
        store (``get`` handles eviction).

        Returns:
            A ``{key: value}`` mapping of every entry whose TTL has not
            elapsed (including entries with no TTL).
        """
        now = time.time()
        return {
            k: v for k, (v, ttl, ts) in self._data.items()
            if ttl is None or now - ts < ttl
        }


@dataclass
class SlowMemory:
    """Append-only episodic log backed by SQLite.

    Every event the runtime chooses to persist (state changes, responses,
    notes) becomes one row in ``episodes``. The schema is deliberately small
    — a timestamp, a ``kind`` tag, the originating actor, a JSON payload,
    and an optional ``trace_id`` — so that downstream consolidators and the
    self-model can scan the log cheaply.

    Attributes:
        db_path: Filesystem path to the SQLite database. The parent
            directory is created on init if it does not exist.
    """
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
        """Append one episode row to the log and commit.

        Args:
            kind: Category tag (e.g. ``"state"``, ``"response"``, ``"note"``).
            actor: Originator of the event (subsystem or user identifier).
            payload: Arbitrary JSON-serialisable dict. Non-JSON values are
                stringified via ``default=str``.
            trace_id: Optional correlation ID linking this row to other
                events from the same turn.
        """
        assert self._conn is not None
        self._conn.execute(
            "INSERT INTO episodes (ts, kind, actor, payload, trace_id) VALUES (?, ?, ?, ?, ?)",
            (time.time(), kind, actor, json.dumps(payload, default=str), trace_id),
        )
        self._conn.commit()

    def query(self, kind: str | None = None, since: float | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Return recent episodes, optionally filtered by kind and time.

        Rows come back newest-first. The ``payload`` column is JSON-decoded
        back into a dict for each result.

        Args:
            kind: If set, only episodes with this ``kind`` tag are returned.
            since: If set, only episodes with ``ts >= since`` are returned.
            limit: Maximum number of rows (most recent first).

        Returns:
            A list of dicts with keys ``ts``, ``kind``, ``actor``,
            ``payload``, ``trace_id`` — one per row, newest first.
        """
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

    def summarize(self, kind: str | None = None, since: float | None = None) -> dict[str, Any]:
        """Aggregate stats over episodes.

        Uses SQL ``GROUP BY`` over the existing connection — cheap even on
        large logs. The consolidator consumes this to build daily digests.

        Args:
            kind: If set, restrict aggregation to rows with this ``kind`` tag.
            since: If set, restrict aggregation to rows with ``ts >= since``.

        Returns:
            A dict with keys:
            - ``total`` (int): number of matching rows.
            - ``by_kind`` (dict[str, int]): count per ``kind`` value.
            - ``by_actor`` (dict[str, int]): count per ``actor`` value.
            - ``first_ts`` (float | None): earliest ``ts`` in the window,
                or ``None`` when empty.
            - ``last_ts`` (float | None): latest ``ts`` in the window, or
                ``None`` when empty.
        """
        assert self._conn is not None
        where = " WHERE 1=1"
        args: list[Any] = []
        if kind:
            where += " AND kind = ?"
            args.append(kind)
        if since:
            where += " AND ts >= ?"
            args.append(since)

        total_row = self._conn.execute(
            f"SELECT COUNT(*), MIN(ts), MAX(ts) FROM episodes{where}", args
        ).fetchone()
        total = int(total_row[0]) if total_row and total_row[0] is not None else 0
        first_ts = total_row[1] if total_row and total_row[1] is not None else None
        last_ts = total_row[2] if total_row and total_row[2] is not None else None

        by_kind: dict[str, int] = {}
        for k, c in self._conn.execute(
            f"SELECT kind, COUNT(*) FROM episodes{where} GROUP BY kind", args
        ).fetchall():
            by_kind[k] = int(c)

        by_actor: dict[str, int] = {}
        for a, c in self._conn.execute(
            f"SELECT actor, COUNT(*) FROM episodes{where} GROUP BY actor", args
        ).fetchall():
            by_actor[a] = int(c)

        return {
            "total": total,
            "by_kind": by_kind,
            "by_actor": by_actor,
            "first_ts": first_ts,
            "last_ts": last_ts,
        }

    def close(self) -> None:
        """Close the SQLite connection. Safe to call more than once."""
        if self._conn:
            self._conn.close()
            self._conn = None
