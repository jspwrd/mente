"""Tests for the CLI dispatch surface.

We don't exercise the interactive REPL; we verify build_parser() is sane,
that `main(["reset"])` wipes .mente* directories, that `main(["test"])`
runs the smoke tests and returns 0, and that --help exits cleanly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from mente import cli


def test_build_parser_has_expected_subcommands() -> None:
    parser = cli.build_parser()
    assert isinstance(parser, argparse.ArgumentParser)

    # Drive the parser with the known commands; if any subcommand is wired
    # incorrectly, parse_args raises SystemExit.
    for cmd in ["run", "demo", "federated", "peer", "test", "reset"]:
        ns = parser.parse_args([cmd])
        assert ns.command == cmd

    ns = parser.parse_args(["federated", "--port", "7903"])
    assert ns.port == 7903

    ns = parser.parse_args(["peer", "--port", "7777", "--id", "peer.extra"])
    assert ns.port == 7777 and ns.id == "peer.extra"


def test_parser_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    # argparse uses 0 on --help
    assert excinfo.value.code == 0


def test_main_reset_removes_mente_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Redirect _root() to the tmp area so reset only touches the sandbox.
    monkeypatch.setattr(cli, "_root", lambda: tmp_path)

    (tmp_path / ".mente").mkdir()
    (tmp_path / ".mente" / "latent.json").write_text("{}")
    (tmp_path / ".mente-test").mkdir()
    (tmp_path / ".mente-hub").mkdir()
    (tmp_path / "not-mente").mkdir()

    rc = cli.main(["reset"])
    assert rc == 0

    assert not (tmp_path / ".mente").exists()
    assert not (tmp_path / ".mente-test").exists()
    assert not (tmp_path / ".mente-hub").exists()
    # A non-.mente-prefixed dir must survive.
    assert (tmp_path / "not-mente").exists()


def test_main_reset_with_nothing_to_remove_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_root", lambda: tmp_path)
    rc = cli.main(["reset"])
    assert rc == 0


def test_main_test_runs_smoke_tests_returns_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The CLI's smoke path writes/uses .mente-test under _root(); point that at
    # tmp_path so we don't stomp on anything in the worktree.
    monkeypatch.setattr(cli, "_root", lambda: tmp_path)

    rc = cli.main(["test"])
    assert rc == 0

    # _smoke_tests() cleans up .mente-test on success; confirm.
    assert not (tmp_path / ".mente-test").exists()


def test_main_unknown_subcommand_exits_nonzero() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["does-not-exist"])
    assert excinfo.value.code != 0


# ---------------------------------------------------------------------------
# migrate
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_migrate_subcommand_appears_in_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    assert "migrate" in out


def test_migrate_subcommand_has_its_own_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["migrate", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--data-dir" in out
    assert "--dry-run" in out


def test_migrate_dry_run_does_not_modify_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force a schema version so v0 files are considered "needs upgrade" — we
    # want to exercise the "would upgrade" code path, not the no-op path.
    monkeypatch.setattr(cli, "_current_schema_version", lambda: 1)
    monkeypatch.setattr(cli, "_root", lambda: tmp_path)
    data = tmp_path / ".mente"
    latent = data / "latent.json"
    _write_json(latent, {"hello": "world"})
    original = latent.read_bytes()

    rc = cli.main(["migrate", "--data-dir", ".mente", "--dry-run"])
    assert rc == 0
    assert latent.read_bytes() == original


def test_migrate_upgrades_then_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "_current_schema_version", lambda: 1)
    monkeypatch.setattr(cli, "_root", lambda: tmp_path)
    data = tmp_path / ".mente"
    latent = data / "latent.json"
    _write_json(latent, {"hello": "world"})  # pre-v1 bare dict

    rc1 = cli.main(["migrate", "--data-dir", ".mente"])
    assert rc1 == 0
    after_first = latent.read_bytes()
    # First pass either upgraded via state._migrate (if present) or stamped
    # _schema via the fallback; either way the second pass must be a no-op.

    rc2 = cli.main(["migrate", "--data-dir", ".mente"])
    assert rc2 == 0
    assert latent.read_bytes() == after_first


def test_migrate_skips_corrupt_files_without_crashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "_root", lambda: tmp_path)
    data = tmp_path / ".mente"
    data.mkdir()
    corrupt = data / "latent.json"
    corrupt.write_text("not valid json {{{")

    rc = cli.main(["migrate", "--data-dir", ".mente"])
    assert rc == 0
    # File still there, still corrupt — migrator warned, didn't delete.
    assert corrupt.read_text() == "not valid json {{{"
    out = capsys.readouterr().out
    assert "skip" in out
    assert "1 skipped" in out


def test_migrate_ignores_files_outside_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "_root", lambda: tmp_path)
    data = tmp_path / ".mente"
    _write_json(data / "latent.json", {"a": 1})
    outside = tmp_path / "not-mente" / "other.json"
    _write_json(outside, {"untouched": True})
    before = outside.read_bytes()

    rc = cli.main(["migrate", "--data-dir", ".mente"])
    assert rc == 0
    assert outside.read_bytes() == before
    out = capsys.readouterr().out
    # Exactly one file inspected — the one inside .mente/.
    assert "1 files inspected" in out


def test_migrate_missing_data_dir_reports_and_returns_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "_root", lambda: tmp_path)
    rc = cli.main(["migrate", "--data-dir", ".mente"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "does not exist" in out


def test_migrate_stamps_unknown_shape_to_target_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Files that don't match a known shape get their _schema marker stamped."""
    monkeypatch.setattr(cli, "_current_schema_version", lambda: 7)
    monkeypatch.setattr(cli, "_root", lambda: tmp_path)
    data = tmp_path / ".mente"
    odd = data / "custom.json"
    # Already-versioned unknown shape; needs to bump from v1 to v7.
    _write_json(odd, {"_schema": 1, "extra": "data"})

    rc = cli.main(["migrate", "--data-dir", ".mente"])
    assert rc == 0
    payload = json.loads(odd.read_text())
    assert payload["_schema"] == 7
    assert payload["extra"] == "data"

    # Second pass — already current — is a no-op.
    before = odd.read_bytes()
    rc2 = cli.main(["migrate", "--data-dir", ".mente"])
    assert rc2 == 0
    assert odd.read_bytes() == before


def test_migrate_no_schema_version_defined_reports_all_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Before Unit 10 lands, _SCHEMA_VERSION is absent; nothing to upgrade."""
    monkeypatch.setattr(cli, "_current_schema_version", lambda: None)
    monkeypatch.setattr(cli, "_root", lambda: tmp_path)
    data = tmp_path / ".mente"
    _write_json(data / "latent.json", {"hello": "world"})

    rc = cli.main(["migrate", "--data-dir", ".mente"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "1 files inspected" in out
    assert "0 upgraded" in out
    assert "1 already current" in out
