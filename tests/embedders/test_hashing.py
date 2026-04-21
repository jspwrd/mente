"""Tests for HashEmbedder (offline fallback)."""
from __future__ import annotations

import math

from mente.embedders import Embedder, HashEmbedder
from mente.embedders.hashing import HashEmbedder as HashingModuleHashEmbedder
from mente.embeddings import HashEmbedder as ShimHashEmbedder


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_hash_embedder_default_dim() -> None:
    emb = HashEmbedder()
    assert emb.dim == 256
    assert emb.ngram_sizes == (3, 4)


def test_hash_embedder_is_embedder_protocol() -> None:
    emb = HashEmbedder()
    # Structural typing check: duck types as Embedder.
    assert hasattr(emb, "dim") and hasattr(emb, "embed")
    e: Embedder = emb
    _ = e.embed("hello")


def test_hash_embedder_output_shape_and_unit_norm() -> None:
    emb = HashEmbedder()
    v = emb.embed("hello world")
    assert len(v) == 256
    norm = math.sqrt(sum(x * x for x in v))
    assert abs(norm - 1.0) < 1e-6


def test_hash_embedder_empty_string_is_zero_vector() -> None:
    emb = HashEmbedder()
    v = emb.embed("")
    assert v == [0.0] * 256


def test_hash_embedder_is_deterministic() -> None:
    a = HashEmbedder().embed("deploy the service")
    b = HashEmbedder().embed("deploy the service")
    assert a == b


def test_hash_embedder_cosine_ordering_preserved() -> None:
    emb = HashEmbedder()
    q = emb.embed("redis uses AOF for durability")
    close = emb.embed("redis AOF durability notes")
    far = emb.embed("baking sourdough bread at home")
    assert _cosine(q, close) > _cosine(q, far)


def test_shim_reexports_same_class() -> None:
    # Backward-compat: mente.embeddings.HashEmbedder must be the same class
    # as the one in mente.embedders.hashing.
    assert ShimHashEmbedder is HashingModuleHashEmbedder
    assert ShimHashEmbedder is HashEmbedder


def test_shim_and_direct_produce_identical_vectors() -> None:
    a = ShimHashEmbedder().embed("semantic memory test")
    b = HashEmbedder().embed("semantic memory test")
    assert a == b


def test_class_hashembedder_not_defined_in_embeddings_module() -> None:
    """Guard against someone reintroducing a body-level class HashEmbedder
    in mente.embeddings; it must stay a re-export."""
    import inspect

    import mente.embeddings as shim
    src = inspect.getsource(shim)
    assert "class HashEmbedder" not in src
