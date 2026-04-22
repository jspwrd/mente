"""Persistent latent state.

This is the mechanism that makes the system a *process*, not a function.
The latent survives turns, sessions, and restarts.

In a real system this would be a learned recurrent hidden state (RWKV/Mamba/Titans).
Here it's a typed dict with JSON serialization — the interface is what matters;
the representation swaps in.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


@dataclass
class LatentState:
    values: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    @classmethod
    def load(cls, path: Path) -> LatentState:
        """Load latent state from ``path``; return an empty state on corruption.

        A corrupt latent.json (bad JSON, non-dict payload) should not brick
        runtime construction — we log a warning, start fresh, and let the
        next checkpoint overwrite the bad file.
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
        return cls(values=data, path=path)

    def checkpoint(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.values, default=str, indent=2))
        tmp.replace(self.path)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value

    def update(self, **kwargs: Any) -> None:
        self.values.update(kwargs)
