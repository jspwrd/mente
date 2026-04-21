"""Tests for :mod:`aria.config` — precedence, coercion, and round-trip."""
from __future__ import annotations

from pathlib import Path

import pytest

from aria.config import AriaConfig


# -- defaults ---------------------------------------------------------------


def test_default_has_expected_values() -> None:
    c = AriaConfig.default()
    assert c.data_root == Path(".aria")
    assert c.bus_host == "127.0.0.1"
    assert c.bus_port == 7722
    assert c.bus_role == "inproc"
    assert c.node_id == "aria.local"
    assert c.consolidator_interval_s == 10.0
    assert c.curiosity_interval_s == 3.0
    assert c.curiosity_idle_threshold_s == 5.0
    assert c.verifier_min_confidence == 0.35
    assert c.router_min_confidence == 0.7
    assert c.router_ms_per_conf == 2000.0
    assert c.llm_model == "claude-opus-4-7"
    assert c.llm_effort == "medium"
    assert c.llm_max_tokens == 4096
    assert c.log_level == "INFO"
    assert c.log_json is False


def test_default_is_frozen() -> None:
    c = AriaConfig.default()
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        c.bus_port = 1234  # type: ignore[misc]


# -- TOML round-trip --------------------------------------------------------


def test_from_toml_roundtrip(tmp_path: Path) -> None:
    toml_body = """
    bus_host = "10.0.0.5"
    bus_port = 8000
    bus_role = "hub"
    node_id = "aria.edge"
    consolidator_interval_s = 15.0
    verifier_min_confidence = 0.5
    llm_model = "claude-sonnet"
    log_json = true
    data_root = "/var/aria"
    """
    path = tmp_path / "aria.toml"
    path.write_text(toml_body)

    c = AriaConfig.from_toml(path)

    assert c.bus_host == "10.0.0.5"
    assert c.bus_port == 8000
    assert c.bus_role == "hub"
    assert c.node_id == "aria.edge"
    assert c.consolidator_interval_s == 15.0
    assert c.verifier_min_confidence == 0.5
    assert c.llm_model == "claude-sonnet"
    assert c.log_json is True
    assert c.data_root == Path("/var/aria")
    # unset fields retain their default.
    assert c.router_ms_per_conf == 2000.0


def test_from_toml_accepts_aria_section(tmp_path: Path) -> None:
    path = tmp_path / "aria.toml"
    path.write_text('[aria]\nbus_port = 9001\n')
    c = AriaConfig.from_toml(path)
    assert c.bus_port == 9001


def test_from_toml_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "aria.toml"
    path.write_text('bus_port = 9001\nmystery_knob = "nope"\n')
    with pytest.raises(ValueError, match="unknown config keys"):
        AriaConfig.from_toml(path)


def test_to_toml_str_parses_back_to_same_values() -> None:
    import tomllib

    c = AriaConfig.default()
    parsed = tomllib.loads(c.to_toml_str())
    # Every field survives the round-trip (Path becomes a string in TOML).
    d = c.to_dict()
    assert parsed == d


# -- env overrides ----------------------------------------------------------


def test_from_env_overrides_selected_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARIA_BUS_PORT", "9999")
    monkeypatch.setenv("ARIA_BUS_ROLE", "spoke")
    monkeypatch.setenv("ARIA_VERIFIER_MIN_CONFIDENCE", "0.9")
    monkeypatch.setenv("ARIA_LOG_JSON", "true")
    monkeypatch.setenv("ARIA_LLM_MAX_TOKENS", "8192")

    c = AriaConfig.from_env()

    assert c.bus_port == 9999
    assert c.bus_role == "spoke"
    assert c.verifier_min_confidence == 0.9
    assert c.log_json is True
    assert c.llm_max_tokens == 8192
    # Unset env vars should remain at default.
    assert c.bus_host == "127.0.0.1"
    assert c.node_id == "aria.local"


def test_from_env_with_no_vars_returns_base() -> None:
    # Use a distinctive base config to confirm from_env(base) respects it.
    base = AriaConfig.default()
    c = AriaConfig.from_env(base)
    assert c == base


def test_from_env_applies_over_given_base(monkeypatch: pytest.MonkeyPatch) -> None:
    import dataclasses

    base = dataclasses.replace(AriaConfig.default(), bus_port=5555, log_level="DEBUG")
    monkeypatch.setenv("ARIA_BUS_PORT", "6666")
    c = AriaConfig.from_env(base)
    assert c.bus_port == 6666  # env wins
    assert c.log_level == "DEBUG"  # base preserved where env absent


# -- bool coercion ----------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("true", True), ("True", True), ("TRUE", True),
        ("false", False), ("False", False), ("FALSE", False),
        ("1", True), ("0", False),
        ("yes", True), ("no", False),
        ("on", True), ("off", False),
        ("y", True), ("n", False),
        ("t", True), ("f", False),
    ],
)
def test_bool_coercion(monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool) -> None:
    monkeypatch.setenv("ARIA_LOG_JSON", raw)
    c = AriaConfig.from_env()
    assert c.log_json is expected


def test_bool_coercion_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARIA_LOG_JSON", "maybe")
    with pytest.raises(ValueError, match="invalid bool"):
        AriaConfig.from_env()


# -- int / float / path coercion -------------------------------------------


def test_invalid_int_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARIA_BUS_PORT", "not-a-number")
    with pytest.raises(ValueError, match="invalid int"):
        AriaConfig.from_env()


def test_invalid_float_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARIA_VERIFIER_MIN_CONFIDENCE", "kinda-high")
    with pytest.raises(ValueError, match="invalid float"):
        AriaConfig.from_env()


def test_path_coercion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARIA_DATA_ROOT", ".my-aria")
    c = AriaConfig.from_env()
    assert c.data_root == Path(".my-aria")


# -- load() precedence ------------------------------------------------------


def test_load_precedence_defaults_toml_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    toml_path = tmp_path / "aria.toml"
    toml_path.write_text(
        'bus_port = 8001\n'
        'bus_role = "hub"\n'
        'node_id = "aria.from-toml"\n'
        'log_level = "DEBUG"\n'
    )
    # Env overrides only one TOML-set field and one default-only field.
    monkeypatch.setenv("ARIA_BUS_PORT", "9000")
    monkeypatch.setenv("ARIA_LLM_MAX_TOKENS", "2048")

    c = AriaConfig.load(toml_path)

    # env wins over TOML.
    assert c.bus_port == 9000
    # TOML wins over defaults.
    assert c.bus_role == "hub"
    assert c.node_id == "aria.from-toml"
    assert c.log_level == "DEBUG"
    # env wins over defaults.
    assert c.llm_max_tokens == 2048
    # untouched defaults preserved.
    assert c.bus_host == "127.0.0.1"


def test_load_without_toml_path_returns_defaults_plus_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARIA_NODE_ID", "aria.solo")
    c = AriaConfig.load()
    assert c.node_id == "aria.solo"
    assert c.bus_port == 7722  # default


def test_load_missing_toml_file_is_silent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "nope.toml"
    monkeypatch.setenv("ARIA_BUS_PORT", "1234")
    c = AriaConfig.load(missing)
    assert c.bus_port == 1234
    assert c.bus_host == "127.0.0.1"


# -- to_dict / hashability --------------------------------------------------


def test_to_dict_coerces_path_to_str() -> None:
    c = AriaConfig.default()
    d = c.to_dict()
    assert d["data_root"] == ".aria"
    assert isinstance(d["data_root"], str)


def test_config_is_hashable() -> None:
    a = AriaConfig.default()
    b = AriaConfig.default()
    assert hash(a) == hash(b)
    assert {a, b} == {a}
