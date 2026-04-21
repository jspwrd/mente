"""Tests for LocalEmbedder (sentence-transformers, offline but real model).

The sentence-transformers package is heavy (~PyTorch), so these tests mock
the SentenceTransformer import unless ``ST_INTEGRATION`` is set.
"""
from __future__ import annotations

import os
import sys
import types

import pytest


def _inject_fake_st(monkeypatch: pytest.MonkeyPatch, vec: list[float], dim: int | None = None) -> None:
    """Install a fake ``sentence_transformers`` module in sys.modules."""
    fake_module = types.ModuleType("sentence_transformers")

    class _FakeST:
        def __init__(self, name: str) -> None:
            self.name = name

        def get_sentence_embedding_dimension(self) -> int:
            return dim if dim is not None else len(vec)

        def encode(self, text, convert_to_numpy: bool = True):  # type: ignore[no-untyped-def]
            if isinstance(text, list):
                return [list(vec) for _ in text]
            return list(vec)

    fake_module.SentenceTransformer = _FakeST  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)


def test_local_embedder_import_error_when_package_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure it's NOT in sys.modules and block new imports.
    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # type: ignore[arg-type]
    from mente.embedders import LocalEmbedder
    with pytest.raises(ImportError, match="sentence-transformers"):
        LocalEmbedder()


def test_local_embedder_returns_unit_normalized_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    _inject_fake_st(monkeypatch, vec=[3.0, 4.0])
    from mente.embedders import LocalEmbedder
    emb = LocalEmbedder(model="fake")
    v = emb.embed("anything")
    # 3-4-5 triangle → normalized to 0.6, 0.8.
    assert v == pytest.approx([0.6, 0.8])
    assert emb.dim == 2


def test_local_embedder_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    _inject_fake_st(monkeypatch, vec=[1.0, 0.0, 0.0])
    from mente.embedders import LocalEmbedder
    emb = LocalEmbedder(model="fake")

    original_encode = emb._model.encode

    def counting_encode(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return original_encode(*args, **kwargs)

    emb._model.encode = counting_encode  # type: ignore[method-assign]
    _ = emb.embed("same text")
    _ = emb.embed("same text")
    assert calls["n"] == 1, "cache hit should skip the second encode()"


def test_local_embedder_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    _inject_fake_st(monkeypatch, vec=[1.0, 0.0, 0.0])
    from mente.embedders import LocalEmbedder
    emb = LocalEmbedder(model="fake")
    out = emb.embed_batch(["a", "b", "a"])  # third one should hit cache
    assert len(out) == 3
    assert all(v == pytest.approx([1.0, 0.0, 0.0]) for v in out)


@pytest.mark.skipif(
    not os.environ.get("ST_INTEGRATION"),
    reason="set ST_INTEGRATION=1 to download a real model and test it",
)
def test_local_embedder_live() -> None:
    from mente.embedders import LocalEmbedder
    emb = LocalEmbedder()
    v = emb.embed("hello world")
    assert len(v) == emb.dim
