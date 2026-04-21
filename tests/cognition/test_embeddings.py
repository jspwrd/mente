"""Tests for mente.embeddings — HashEmbedder determinism/normalization + SemanticMemory."""
from __future__ import annotations

import math

from mente.embeddings import HashEmbedder, SemanticMemory, _cosine


def test_hash_embedder_deterministic():
    e = HashEmbedder()
    v1 = e.embed("the quick brown fox")
    v2 = e.embed("the quick brown fox")
    assert v1 == v2


def test_hash_embedder_unit_normalized():
    e = HashEmbedder()
    for text in ["hello", "deployment pipeline", "one two three four"]:
        v = e.embed(text)
        norm = math.sqrt(sum(x * x for x in v))
        assert math.isclose(norm, 1.0, abs_tol=1e-6)


def test_hash_embedder_dim_respects_config():
    e = HashEmbedder(dim=64)
    v = e.embed("hello")
    assert len(v) == 64


def test_hash_embedder_empty_input_returns_zero_vector():
    e = HashEmbedder()
    v = e.embed("   !! ")  # no alnum tokens
    assert all(x == 0.0 for x in v)


def test_ngram_makes_deploy_cluster_with_deployment():
    e = HashEmbedder()
    deploy = e.embed("deploy")
    deployment = e.embed("deployment")
    unrelated = e.embed("banana")
    assert _cosine(deploy, deployment) > _cosine(deploy, unrelated)
    assert _cosine(deploy, deployment) > 0.5


def test_ngram_similar_roots_cluster():
    e = HashEmbedder()
    run = e.embed("running")
    ran = e.embed("runner")
    elephant = e.embed("elephant")
    assert _cosine(run, ran) > _cosine(run, elephant)


def test_cosine_of_identical_vectors_is_one():
    e = HashEmbedder()
    v = e.embed("hello world")
    assert math.isclose(_cosine(v, v), 1.0, abs_tol=1e-6)


def test_semantic_memory_remember_and_search_returns_sorted_hits(tmp_path):
    mem = SemanticMemory(db_path=tmp_path / "m.db")
    try:
        mem.remember("redis uses AOF or RDB for persistence")
        mem.remember("postgres uses WAL for durability")
        mem.remember("bananas are yellow")
        hits = mem.search("redis persistence", k=3)
        assert len(hits) == 3
        # Descending by score.
        assert hits[0]["score"] >= hits[1]["score"] >= hits[2]["score"]
        # Top hit should be the redis document.
        assert "redis" in hits[0]["text"].lower()
    finally:
        mem.close()


def test_semantic_memory_k_limits_results(tmp_path):
    mem = SemanticMemory(db_path=tmp_path / "m.db")
    try:
        for i in range(5):
            mem.remember(f"note {i}")
        hits = mem.search("note", k=2)
        assert len(hits) == 2
    finally:
        mem.close()


def test_semantic_memory_kind_filter(tmp_path):
    mem = SemanticMemory(db_path=tmp_path / "m.db")
    try:
        mem.remember("a note", kind="note")
        mem.remember("a thought", kind="thought")
        hits = mem.search("a", kind="note")
        assert all(h["kind"] == "note" for h in hits)
    finally:
        mem.close()


def test_semantic_memory_persists_across_instances(tmp_path):
    db = tmp_path / "m.db"
    mem1 = SemanticMemory(db_path=db)
    try:
        mem1.remember("hello world")
    finally:
        mem1.close()
    mem2 = SemanticMemory(db_path=db)
    try:
        hits = mem2.search("hello", k=1)
        assert hits and "hello world" in hits[0]["text"]
    finally:
        mem2.close()
