"""Tests for the CLI dispatch surface.

We don't exercise the interactive REPL; we verify build_parser() is sane,
that `main(["reset"])` wipes .aria* directories, that `main(["test"])`
runs the smoke tests and returns 0, and that --help exits cleanly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from aria import cli


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


def test_main_reset_removes_aria_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Redirect _root() to the tmp area so reset only touches the sandbox.
    monkeypatch.setattr(cli, "_root", lambda: tmp_path)

    (tmp_path / ".aria").mkdir()
    (tmp_path / ".aria" / "latent.json").write_text("{}")
    (tmp_path / ".aria-test").mkdir()
    (tmp_path / ".aria-hub").mkdir()
    (tmp_path / "not-aria").mkdir()

    rc = cli.main(["reset"])
    assert rc == 0

    assert not (tmp_path / ".aria").exists()
    assert not (tmp_path / ".aria-test").exists()
    assert not (tmp_path / ".aria-hub").exists()
    # A non-.aria-prefixed dir must survive.
    assert (tmp_path / "not-aria").exists()


def test_main_reset_with_nothing_to_remove_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_root", lambda: tmp_path)
    rc = cli.main(["reset"])
    assert rc == 0


def test_main_test_runs_smoke_tests_returns_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The CLI's smoke path writes/uses .aria-test under _root(); point that at
    # tmp_path so we don't stomp on anything in the worktree.
    monkeypatch.setattr(cli, "_root", lambda: tmp_path)

    rc = cli.main(["test"])
    assert rc == 0

    # _smoke_tests() cleans up .aria-test on success; confirm.
    assert not (tmp_path / ".aria-test").exists()


def test_main_unknown_subcommand_exits_nonzero() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["does-not-exist"])
    assert excinfo.value.code != 0
