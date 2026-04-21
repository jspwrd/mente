"""Single source of truth for MENTE runtime configuration.

Precedence at load time: defaults < TOML file < environment variables.

This module is intentionally stdlib-only and imports no other MENTE module.
Adoption by runtime/cli/etc. happens in a follow-up unit; for now this is a
future-adoption seed that launch-time ops can exercise and tests can cover.

Env-var naming: every dataclass field maps to ``MENTE_<FIELD_UPPER>``. For
example, ``bus_port`` -> ``MENTE_BUS_PORT`` and ``verifier_min_confidence``
-> ``MENTE_VERIFIER_MIN_CONFIDENCE``.
"""
from __future__ import annotations

import dataclasses
import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


_TRUE_STRINGS = {"1", "true", "yes", "on", "y", "t"}
_FALSE_STRINGS = {"0", "false", "no", "off", "n", "f"}

# ``from __future__ import annotations`` makes ``dataclasses.Field.type`` a
# string. Map those strings back to real types for coercion without pulling in
# ``typing.get_type_hints`` (which would try to resolve the whole module).
_TYPE_MAP: dict[str, type] = {
    "bool": bool,
    "int": int,
    "float": float,
    "str": str,
    "Path": Path,
}


def _resolve_type(annotation: Any) -> type:
    """Turn a dataclass field ``type`` (possibly a string) into a real type."""
    if isinstance(annotation, str):
        try:
            return _TYPE_MAP[annotation]
        except KeyError as e:
            raise ValueError(f"unsupported type annotation: {annotation!r}") from e
    return annotation


def _coerce(value: str, annotation: Any, field_name: str) -> Any:
    """Coerce a string (typically from env) into the dataclass field's type.

    Raises ``ValueError`` with a clear message if coercion fails.
    """
    target_type = _resolve_type(annotation)
    if target_type is bool:
        lowered = value.strip().lower()
        if lowered in _TRUE_STRINGS:
            return True
        if lowered in _FALSE_STRINGS:
            return False
        raise ValueError(
            f"invalid bool for {field_name!r}: {value!r} "
            f"(expected one of true/false/1/0/yes/no/on/off)"
        )
    if target_type is int:
        try:
            return int(value)
        except ValueError as e:
            raise ValueError(f"invalid int for {field_name!r}: {value!r}") from e
    if target_type is float:
        try:
            return float(value)
        except ValueError as e:
            raise ValueError(f"invalid float for {field_name!r}: {value!r}") from e
    if target_type is Path:
        return Path(value)
    if target_type is str:
        return value
    raise ValueError(f"unsupported type {target_type!r} for field {field_name!r}")


def _env_var_name(field_name: str) -> str:
    return f"MENTE_{field_name.upper()}"


@dataclass(frozen=True)
class MenteConfig:
    """Immutable runtime configuration for an MENTE process."""

    data_root: Path = Path(".mente")
    bus_host: str = "127.0.0.1"
    bus_port: int = 7722
    bus_role: str = "inproc"  # one of: "inproc", "hub", "spoke"
    node_id: str = "mente.local"
    consolidator_interval_s: float = 10.0
    curiosity_interval_s: float = 3.0
    curiosity_idle_threshold_s: float = 5.0
    verifier_min_confidence: float = 0.35
    router_min_confidence: float = 0.7
    router_ms_per_conf: float = 2000.0
    llm_model: str = "claude-opus-4-7"
    llm_effort: str = "medium"
    llm_max_tokens: int = 4096
    log_level: str = "INFO"
    log_json: bool = False

    # -- constructors -------------------------------------------------------
    @classmethod
    def default(cls) -> "MenteConfig":
        """Return a config populated entirely with the declared defaults."""
        return cls()

    @classmethod
    def from_toml(cls, path: Path | str) -> "MenteConfig":
        """Parse a TOML file into an ``MenteConfig``.

        Unknown keys raise ``ValueError`` so typos surface loudly. Values are
        coerced per the field's annotation where the TOML type doesn't match
        directly (e.g. a string for ``data_root`` becomes a ``Path``).
        """
        toml_path = Path(path)
        with toml_path.open("rb") as fh:
            data = tomllib.load(fh)
        # Support either a flat table or an [mente] section.
        if "mente" in data and isinstance(data["mente"], dict):
            data = data["mente"]
        return cls._apply_overrides(cls.default(), data, source=str(toml_path))

    @classmethod
    def from_env(cls, base: "MenteConfig | None" = None) -> "MenteConfig":
        """Return ``base`` (or defaults) with any ``MENTE_*`` env vars applied."""
        base = base if base is not None else cls.default()
        overrides: dict[str, Any] = {}
        for f in fields(cls):
            raw = os.environ.get(_env_var_name(f.name))
            if raw is None:
                continue
            overrides[f.name] = _coerce(raw, f.type, f.name)
        return dataclasses.replace(base, **overrides)

    @classmethod
    def load(cls, toml_path: Path | str | None = None) -> "MenteConfig":
        """Single entry point: defaults -> TOML (if present) -> env overrides.

        If ``toml_path`` is given but the file doesn't exist, TOML is silently
        skipped so callers can point at an optional config without guarding.
        """
        cfg = cls.default()
        if toml_path is not None:
            try:
                cfg = cls.from_toml(toml_path)
            except FileNotFoundError:
                pass
        return cls.from_env(cfg)

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view with ``Path`` coerced to ``str`` for JSON/TOML."""
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Path):
                value = str(value)
            out[f.name] = value
        return out

    def to_toml_str(self) -> str:
        """Render the config as TOML text. Stdlib has no writer, so emit by hand."""
        lines: list[str] = []
        for f in fields(self):
            value = getattr(self, f.name)
            lines.append(f"{f.name} = {_toml_value(value)}")
        return "\n".join(lines) + "\n"

    # -- internal -----------------------------------------------------------
    @classmethod
    def _apply_overrides(
        cls,
        base: "MenteConfig",
        overrides: dict[str, Any],
        *,
        source: str,
    ) -> "MenteConfig":
        known = {f.name: f for f in fields(cls)}
        unknown = sorted(set(overrides) - known.keys())
        if unknown:
            raise ValueError(
                f"unknown config keys in {source}: {', '.join(unknown)}"
            )
        coerced: dict[str, Any] = {}
        for name, value in overrides.items():
            target = _resolve_type(known[name].type)
            if isinstance(value, str) and target is not str:
                coerced[name] = _coerce(value, target, name)
            elif target is Path and not isinstance(value, Path):
                coerced[name] = Path(value)
            elif target is float and isinstance(value, int) and not isinstance(value, bool):
                coerced[name] = float(value)
            else:
                coerced[name] = value
        return dataclasses.replace(base, **coerced)


def _toml_value(value: Any) -> str:
    """Render a Python value as a TOML literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    # Strings, Paths, anything else -> quoted string.
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'
