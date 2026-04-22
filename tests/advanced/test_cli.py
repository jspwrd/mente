"""Tests for the CLI dispatch surface.

We don't exercise the interactive REPL; we verify build_parser() is sane,
that `main(["reset"])` wipes .mente* directories, that `main(["test"])`
runs the smoke tests and returns 0, and that --help exits cleanly.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import types
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
# train-verifier
# ---------------------------------------------------------------------------


def test_train_verifier_appears_in_top_level_help() -> None:
    """The subcommand must be listed in `mente --help`."""
    parser = cli.build_parser()
    help_text = parser.format_help()
    assert "train-verifier" in help_text


def test_train_verifier_subparser_parses() -> None:
    parser = cli.build_parser()
    ns = parser.parse_args(["train-verifier"])
    assert ns.command == "train-verifier"
    assert ns.data_dir == ".mente"
    assert ns.output == "verifier.joblib"
    assert ns.min_samples == 50

    ns = parser.parse_args([
        "train-verifier", "--data-dir", "custom", "--output", "/tmp/x.joblib",
        "--min-samples", "10",
    ])
    assert ns.data_dir == "custom"
    assert ns.output == "/tmp/x.joblib"
    assert ns.min_samples == 10


def test_train_verifier_subparser_own_help_works() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["train-verifier", "--help"])
    assert excinfo.value.code == 0


def test_train_verifier_missing_extras_prints_install_hint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When sklearn/joblib/baseline can't be imported, we exit 1 cleanly.

    We simulate the verifier-ml extra being absent by poisoning the relevant
    entries in ``sys.modules`` with ``None`` so Python raises
    ``ModuleNotFoundError`` on import. Works whether or not the real packages
    are installed in the test env.
    """
    monkeypatch.setattr(cli, "_root", lambda: tmp_path)

    # sys.modules[name] = None makes `import name` raise ModuleNotFoundError.
    monkeypatch.setitem(sys.modules, "mente.verifiers.baseline", None)
    monkeypatch.setitem(sys.modules, "joblib", None)

    rc = cli.main(["train-verifier", "--data-dir", ".mente"])
    assert rc == 1

    captured = capsys.readouterr()
    assert "mente[verifier-ml] not installed" in captured.err
    assert "pip install 'mente[verifier-ml]'" in captured.err


def test_train_verifier_missing_episodic_db_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """If SlowMemory's SQLite file is absent we fail with a clear message."""
    monkeypatch.setattr(cli, "_root", lambda: tmp_path)

    # Stub the baseline + joblib imports so we reach the db-check path. We
    # inject modules with a harmless ``train_baseline`` / ``dump`` before
    # calling cli.main so that the defensive imports succeed.
    fake_baseline = types.ModuleType("mente.verifiers.baseline")
    fake_baseline.train_baseline = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mente.verifiers.baseline", fake_baseline)

    fake_joblib = types.ModuleType("joblib")
    fake_joblib.dump = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "joblib", fake_joblib)

    rc = cli.main(["train-verifier", "--data-dir", ".does-not-exist"])
    assert rc == 1
    captured = capsys.readouterr()
    assert "no episodic log" in captured.err


# Gate the real-training path on both sklearn and unit-5's baseline module.
_has_sklearn = importlib.util.find_spec("sklearn") is not None
_has_baseline = importlib.util.find_spec("mente.verifiers.baseline") is not None


@pytest.mark.skipif(
    not (_has_sklearn and _has_baseline),
    reason="requires sklearn and mente.verifiers.baseline (unit 5)",
)
def test_train_verifier_writes_output_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: populated SlowMemory -> trained scorer persisted to disk.

    This runs only when ``sklearn`` is installed and unit 5's baseline
    module is importable. Otherwise the test is skipped.
    """
    pytest.importorskip("sklearn")
    pytest.importorskip("joblib")

    monkeypatch.setattr(cli, "_root", lambda: tmp_path)

    # Seed an episodic SQLite with enough labelled rows for training.
    from mente.memory import SlowMemory

    data_dir = tmp_path / ".mente"
    data_dir.mkdir(parents=True, exist_ok=True)
    slow = SlowMemory(db_path=data_dir / "episodic.sqlite")
    for i in range(60):
        slow.record(
            kind="verifier_label",
            actor="test",
            payload={
                "accepted": bool(i % 2),
                "features": {
                    "response_len": float(10 + i),
                    "confidence": 0.5 + (i % 3) * 0.1,
                    "tool_count": float(i % 2),
                },
            },
        )
    slow.close()

    output = tmp_path / "verifier.joblib"
    rc = cli.main([
        "train-verifier",
        "--data-dir", ".mente",
        "--output", str(output),
        "--min-samples", "10",
    ])

    assert rc == 0, "baseline module is importable but training failed"
    assert output.exists()
