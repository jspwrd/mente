"""Tests for VoyageEmbedder.

voyageai is not installed in CI; the ImportError path is exercised by
the real import failure, while the happy path uses a mocked Client.
"""
from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock

import pytest


def _ensure_voyage_absent() -> None:
    sys.modules.pop("voyageai", None)
    sys.modules.pop("mente.embedders.voyage", None)


def _install_fake_voyageai(embeddings_vec: list[float]) -> MagicMock:
    """Install a fake `voyageai` module in sys.modules and return the
    MagicMock stand-in for `voyageai.Client` (so individual tests can
    inspect call counts)."""
    fake = types.ModuleType("voyageai")
    client_instance = MagicMock()
    result = MagicMock()
    result.embeddings = [list(embeddings_vec)]
    client_instance.embed.return_value = result
    client_class = MagicMock(return_value=client_instance)
    fake.Client = client_class  # type: ignore[attr-defined]
    sys.modules["voyageai"] = fake
    # Force re-import so the lazy `import voyageai` inside __init__ sees
    # the fake.
    sys.modules.pop("mente.embedders.voyage", None)
    return client_class


def test_voyage_embedder_raises_clean_import_error_when_missing() -> None:
    _ensure_voyage_absent()
    # A `None` entry in sys.modules makes `import voyageai` raise
    # ImportError — cleanly exercises the missing-dep path without
    # depending on whether voyageai is actually installed.
    sys.modules["voyageai"] = None  # type: ignore[assignment]
    try:
        sys.modules.pop("mente.embedders.voyage", None)
        from mente.embedders.voyage import VoyageEmbedder

        with pytest.raises(ImportError) as excinfo:
            VoyageEmbedder(api_key="fake")
        msg = str(excinfo.value)
        assert "voyageai" in msg
        assert "pip install" in msg
    finally:
        sys.modules.pop("voyageai", None)
        sys.modules.pop("mente.embedders.voyage", None)


def test_voyage_embedder_embed_returns_canned_vector() -> None:
    canned = [0.1, 0.2, 0.3, 0.4]
    try:
        client_class = _install_fake_voyageai(canned)
        from mente.embedders.voyage import VoyageEmbedder

        emb = VoyageEmbedder(api_key="fake-key", model="voyage-3", dim=4)
        out = emb.embed("hello")
        assert out == canned
        # Client called with api_key kwarg because we passed one.
        client_class.assert_called_once_with(api_key="fake-key")
        client_instance = client_class.return_value
        client_instance.embed.assert_called_once_with(["hello"], model="voyage-3")
    finally:
        sys.modules.pop("voyageai", None)
        sys.modules.pop("mente.embedders.voyage", None)


def test_voyage_embedder_lru_cache_hit_skips_client() -> None:
    canned = [1.0, 0.0, 0.0, 0.0]
    try:
        client_class = _install_fake_voyageai(canned)
        from mente.embedders.voyage import VoyageEmbedder

        emb = VoyageEmbedder(api_key="fake-key")
        client_instance = client_class.return_value

        v1 = emb.embed("the same text")
        v2 = emb.embed("the same text")
        assert v1 == v2 == canned
        assert client_instance.embed.call_count == 1

        emb.embed("different text")
        assert client_instance.embed.call_count == 2
    finally:
        sys.modules.pop("voyageai", None)
        sys.modules.pop("mente.embedders.voyage", None)


def test_voyage_embedder_falls_back_to_env_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    canned = [0.5, 0.5]
    try:
        client_class = _install_fake_voyageai(canned)
        monkeypatch.setenv("VOYAGE_API_KEY", "from-env")
        from mente.embedders.voyage import VoyageEmbedder

        emb = VoyageEmbedder()
        client_class.assert_called_once_with(api_key="from-env")
        _ = emb.embed("x")
    finally:
        sys.modules.pop("voyageai", None)
        sys.modules.pop("mente.embedders.voyage", None)


def test_voyage_embedder_embed_batch_caches_and_reuses() -> None:
    canned = [9.0, 9.0]
    try:
        client_class = _install_fake_voyageai(canned)
        # embed_batch with multiple texts expects len(embeddings) ==
        # len(input) — rewire the fake so we return a matching list.
        client_instance = client_class.return_value

        def _fake_embed(texts: list[str], model: str) -> MagicMock:
            r = MagicMock()
            r.embeddings = [[float(i), float(i)] for i, _ in enumerate(texts)]
            return r

        client_instance.embed.side_effect = _fake_embed

        from mente.embedders.voyage import VoyageEmbedder

        emb = VoyageEmbedder(api_key="k")
        out = emb.embed_batch(["a", "b"])
        assert out == [[0.0, 0.0], [1.0, 1.0]]
        # Re-requesting cached texts should not hit the client.
        calls_before = client_instance.embed.call_count
        out2 = emb.embed_batch(["a", "b"])
        assert out2 == [[0.0, 0.0], [1.0, 1.0]]
        assert client_instance.embed.call_count == calls_before
    finally:
        sys.modules.pop("voyageai", None)
        sys.modules.pop("mente.embedders.voyage", None)


@pytest.mark.skipif(
    not os.environ.get("VOYAGE_API_KEY"),
    reason="real Voyage API key not configured",
)
def test_voyage_embedder_live_api_smoke() -> None:
    # Only runs when VOYAGE_API_KEY is set and voyageai is installed.
    from mente.embedders.voyage import VoyageEmbedder

    emb = VoyageEmbedder()
    vec = emb.embed("hello world")
    assert isinstance(vec, list)
    assert len(vec) > 0
    assert all(isinstance(x, float) for x in vec)
