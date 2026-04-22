"""Tests for the baseline trained-verifier scorer.

Covers three paths:

1. Too-little-data: the rule-based fallback fires without ever touching
   sklearn, and returns a calibrated-ish probability.
2. Full training run: with a clearly separable synthetic dataset the trained
   scorer discriminates at AUC > 0.75. Gated behind ``pytest.importorskip``
   so CI without the ``verifier-ml`` extra still passes.
3. Missing sklearn: the lazy import emits a clean hint pointing at
   ``mente[verifier-ml]``.
"""
from __future__ import annotations

import builtins
import random
import sys
from pathlib import Path

import pytest

from mente.memory import SlowMemory
from mente.verifiers.baseline import (
    FEATURE_KEYS,
    _rule_based_scorer,
    train_baseline,
)

# -- helpers ----------------------------------------------------------------


def _make_mem(tmp_path: Path) -> SlowMemory:
    return SlowMemory(db_path=tmp_path / "mem.sqlite")


def _record_response(
    mem: SlowMemory,
    *,
    accept: bool,
    confidence: float,
    tier: str = "fast",
    text: str = "hello",
    tools: list[str] | None = None,
    score: float | None = None,
) -> None:
    mem.record(
        kind="response",
        actor="test",
        payload={
            "text": text,
            "tier": tier,
            "tools": tools or [],
            "confidence": confidence,
            "verdict": {
                "accept": accept,
                "score": score if score is not None else (0.9 if accept else 0.2),
                "reasons": ["ok"] if accept else ["low confidence"],
            },
        },
    )


# -- fallback path ----------------------------------------------------------


def test_rule_based_scorer_returns_valid_probability():
    scorer = _rule_based_scorer()
    p = scorer({"confidence": 0.9, "tier_fast": 1.0})
    assert 0.0 < p < 1.0


def test_rule_based_scorer_rewards_high_confidence():
    scorer = _rule_based_scorer()
    low = scorer({"confidence": 0.1, "tier_fast": 1.0})
    high = scorer({"confidence": 0.95, "tier_fast": 1.0})
    assert high > low


def test_train_baseline_falls_back_when_too_few_samples(tmp_path):
    mem = _make_mem(tmp_path)
    for _ in range(5):  # far below default min_samples=50
        _record_response(mem, accept=True, confidence=0.9)

    scorer = train_baseline(mem, min_samples=50)

    # Must produce a valid probability for a canonical feature dict without
    # any sklearn import having happened.
    features = dict.fromkeys(FEATURE_KEYS, 0.0)
    features["confidence"] = 0.9
    features["tier_fast"] = 1.0
    p = scorer(features)
    assert 0.0 < p < 1.0
    mem.close()


def test_train_baseline_falls_back_on_single_class(tmp_path):
    """Single-label corpus cannot be fit — should return the fallback cleanly."""
    mem = _make_mem(tmp_path)
    for _ in range(60):
        _record_response(mem, accept=True, confidence=0.9)

    scorer = train_baseline(mem, min_samples=10)
    # Rule-based fallback still gives a probability.
    p = scorer({"confidence": 0.9, "tier_fast": 1.0})
    assert 0.0 < p < 1.0
    mem.close()


def test_train_baseline_skips_rows_missing_verdict(tmp_path):
    """Payloads without a usable verdict must not crash training."""
    mem = _make_mem(tmp_path)
    mem.record(kind="response", actor="test", payload={"text": "no verdict"})
    mem.record(
        kind="response",
        actor="test",
        payload={"text": "bad verdict", "verdict": {"score": 0.5}},
    )
    scorer = train_baseline(mem, min_samples=50)
    assert 0.0 < scorer({"confidence": 0.5}) < 1.0
    mem.close()


# -- missing-sklearn path ---------------------------------------------------


def test_missing_sklearn_raises_clean_hint(monkeypatch, tmp_path):
    """With sklearn hidden, training against enough data raises a clear error."""
    mem = _make_mem(tmp_path)
    for _ in range(30):
        _record_response(mem, accept=True, confidence=0.9)
    for _ in range(30):
        _record_response(mem, accept=False, confidence=0.2)

    # Force any ``import sklearn[.something]`` to fail — emulates the extra
    # not being installed.
    real_import = builtins.__import__

    def _fake_import(name: str, *args, **kwargs):
        if name == "sklearn" or name.startswith("sklearn."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    # Also nuke any cached sklearn modules so the lazy import actually runs.
    for mod_name in list(sys.modules):
        if mod_name == "sklearn" or mod_name.startswith("sklearn."):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

    with pytest.raises(ImportError, match=r"mente\[verifier-ml\]"):
        train_baseline(mem, min_samples=10)
    mem.close()


# -- trained-model path -----------------------------------------------------


def test_trained_scorer_discriminates(tmp_path):
    """With a separable synthetic dataset the fitted scorer beats chance."""
    pytest.importorskip("sklearn")

    rng = random.Random(1234)
    mem = _make_mem(tmp_path)

    # Accepted rows: high confidence, few tools, short text.
    for _ in range(100):
        _record_response(
            mem,
            accept=True,
            confidence=rng.uniform(0.75, 0.99),
            tier="specialist",
            text="ok" * rng.randint(5, 20),
            tools=[],
            score=rng.uniform(0.8, 0.95),
        )
    # Rejected rows: low confidence, more tools, longer text.
    for _ in range(100):
        _record_response(
            mem,
            accept=False,
            confidence=rng.uniform(0.05, 0.3),
            tier="deep",
            text="bad" * rng.randint(20, 80),
            tools=["tool_a", "tool_b"],
            score=rng.uniform(0.1, 0.3),
        )

    scorer = train_baseline(mem, min_samples=10)

    # Compute AUC on the training set itself — we only need to confirm the
    # model has learned the separable signal. Use a simple Mann-Whitney U
    # implementation so we don't add sklearn.metrics as a hard test dep.
    pos: list[float] = []
    neg: list[float] = []
    rows = mem.query(kind="response", limit=10_000)
    for row in rows:
        v = row["payload"]["verdict"]
        features = {
            "confidence": row["payload"]["confidence"],
            "text_len": len(row["payload"]["text"]),
            "tool_count": len(row["payload"]["tools"]),
            "tier_fast": 1.0 if row["payload"]["tier"] == "fast" else 0.0,
            "tier_specialist": 1.0 if row["payload"]["tier"] == "specialist" else 0.0,
            "tier_deep": 1.0 if row["payload"]["tier"] == "deep" else 0.0,
            "verdict_score": v["score"],
        }
        (pos if v["accept"] else neg).append(scorer(features))

    auc = _auc(pos, neg)
    assert auc > 0.75, f"expected AUC > 0.75, got {auc:.3f}"
    mem.close()


def _auc(pos: list[float], neg: list[float]) -> float:
    """Mann-Whitney-U AUC. No numpy/sklearn dependency."""
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))
