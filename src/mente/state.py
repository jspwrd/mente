"""Persistent latent state.

This is the mechanism that makes the system a *process*, not a function.
The latent survives turns, sessions, and restarts.

In a real system this would be a learned recurrent hidden state (RWKV/Mamba/Titans).
Here it's a typed dict with JSON serialization — the interface is what matters;
the representation swaps in.

On-disk schema
--------------
The serialized form is an envelope::

    {"_schema": <int>, "values": {<key>: <value>, ...}}

``_SCHEMA_VERSION`` is the version ``checkpoint()`` writes today. ``load()``
reads the envelope, dispatches through ``_MIGRATIONS`` to bring older payloads
forward, and starts empty on unknown future versions (forward-incompatible).

Migrations
----------
``_MIGRATIONS[n]`` transforms a v``n`` payload into a v``n+1`` payload. The
initial entry (v0 → v1) wraps a bare ``{key: value}`` dict into the envelope.
Future schema bumps append a new entry and increment ``_SCHEMA_VERSION``.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

_SCHEMA_VERSION: int = 1


def _migrate_v0_to_v1(raw: dict[str, Any]) -> dict[str, Any]:
    """v0 was a bare ``{key: value}`` dict; wrap it in the v1 envelope."""
    return {"_schema": 1, "values": dict(raw)}


_MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {
    0: _migrate_v0_to_v1,
}


@dataclass
class LatentState:
    values: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    @classmethod
    def load(cls, path: Path) -> LatentState:
        """Load latent state from ``path``; return an empty state on corruption.

        A corrupt latent.json (bad JSON, non-dict payload) should not brick
        runtime construction — we log a warning, start fresh, and let the
        next checkpoint overwrite the bad file. Old schema versions are
        migrated forward; unknown future versions start empty (and will be
        overwritten on the next checkpoint).
        """
        if not path.exists():
            return cls(values={}, path=path)
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            _log.warning("latent file %s is corrupt JSON; starting empty: %s", path, e)
            return cls(values={}, path=path)
        if not isinstance(data, dict):
            _log.warning("latent file %s has unexpected shape; starting empty", path)
            return cls(values={}, path=path)

        version = data.get("_schema")
        if version is None:
            # Pre-versioning format: bare {key: value} dict.
            _log.warning(
                "latent file %s is pre-versioning (v0); migrating to v%d",
                path, _SCHEMA_VERSION,
            )
            data = _migrate(data, from_version=0)
        elif not isinstance(version, int):
            _log.warning(
                "latent file %s has non-integer _schema %r; starting empty",
                path, version,
            )
            return cls(values={}, path=path)
        elif version < _SCHEMA_VERSION:
            _log.info(
                "latent file %s is schema v%d; migrating to v%d",
                path, version, _SCHEMA_VERSION,
            )
            data = _migrate(data, from_version=version)
        elif version > _SCHEMA_VERSION:
            _log.warning(
                "latent file %s is schema v%d (newer than supported v%d); starting empty",
                path, version, _SCHEMA_VERSION,
            )
            return cls(values={}, path=path)

        values = data.get("values")
        if not isinstance(values, dict):
            _log.warning("latent file %s envelope missing 'values' dict; starting empty", path)
            return cls(values={}, path=path)
        return cls(values=values, path=path)

    def checkpoint(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {"_schema": _SCHEMA_VERSION, "values": self.values}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(envelope, default=str, indent=2))
        tmp.replace(self.path)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value

    def update(self, **kwargs: Any) -> None:
        self.values.update(kwargs)


def _migrate(data: dict[str, Any], *, from_version: int) -> dict[str, Any]:
    """Run ``data`` through the migration chain up to ``_SCHEMA_VERSION``."""
    current = data
    version = from_version
    while version < _SCHEMA_VERSION:
        step = _MIGRATIONS.get(version)
        if step is None:
            _log.warning(
                "no migration from v%d to v%d; starting empty",
                version, version + 1,
            )
            return {"_schema": _SCHEMA_VERSION, "values": {}}
        current = step(current)
        version += 1
    return current
