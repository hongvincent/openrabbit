"""Tests for the offline embeddings interface + in-memory vector store (item 16).

The vector path is interface-ready but offline-safe: the default
:class:`FakeEmbedder` is deterministic and hashing-based (no network), and a
:class:`BedrockEmbedder` stub exists (lazy, real impl deferred) so the interface
is complete. A tiny :class:`VectorStore` does cosine top-k in memory.

These tests assert: FakeEmbedder determinism + dimension, interface conformance,
cosine top-k correctness, and that the Bedrock stub does not require boto3 to
import (lazy) and refuses to embed in v1.
"""

from __future__ import annotations

import math

import pytest

from openrabbit.index.embeddings import (
    BedrockEmbedder,
    Embedder,
    FakeEmbedder,
    VectorStore,
    cosine_similarity,
)


# --------------------------------------------------------------------------- #
# FakeEmbedder determinism + shape                                              #
# --------------------------------------------------------------------------- #
def test_fake_embedder_is_deterministic() -> None:
    emb = FakeEmbedder(dim=64)
    a = emb.embed(["hello world"])
    b = emb.embed(["hello world"])
    assert a == b


def test_fake_embedder_dimension() -> None:
    emb = FakeEmbedder(dim=32)
    vecs = emb.embed(["x", "y", "z"])
    assert len(vecs) == 3
    assert all(len(v) == 32 for v in vecs)


def test_fake_embedder_different_text_different_vector() -> None:
    emb = FakeEmbedder(dim=64)
    [va] = emb.embed(["alpha"])
    [vb] = emb.embed(["beta"])
    assert va != vb


def test_fake_embedder_is_normalized() -> None:
    emb = FakeEmbedder(dim=48)
    [v] = emb.embed(["normalize me"])
    norm = math.sqrt(sum(x * x for x in v))
    assert pytest.approx(norm, abs=1e-6) == 1.0


def test_fake_embedder_empty_input() -> None:
    emb = FakeEmbedder(dim=16)
    assert emb.embed([]) == []


def test_fake_embedder_empty_string_is_safe() -> None:
    emb = FakeEmbedder(dim=8)
    [v] = emb.embed([""])
    assert len(v) == 8
    # zero-vector is normalized to all-zeros (no div-by-zero crash).
    assert all(x == 0.0 for x in v) or pytest.approx(
        math.sqrt(sum(x * x for x in v))
    ) == 1.0


def test_fake_embedder_conforms_to_interface() -> None:
    assert isinstance(FakeEmbedder(), Embedder)
    assert FakeEmbedder(dim=10).dim == 10


# --------------------------------------------------------------------------- #
# cosine similarity                                                             #
# --------------------------------------------------------------------------- #
def test_cosine_identical_is_one() -> None:
    v = [1.0, 2.0, 3.0]
    assert pytest.approx(cosine_similarity(v, v), abs=1e-9) == 1.0


def test_cosine_orthogonal_is_zero() -> None:
    assert pytest.approx(cosine_similarity([1.0, 0.0], [0.0, 1.0])) == 0.0


def test_cosine_zero_vector_is_zero() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        cosine_similarity([1.0], [1.0, 2.0])


# --------------------------------------------------------------------------- #
# VectorStore top-k                                                             #
# --------------------------------------------------------------------------- #
def test_vector_store_topk_orders_by_similarity() -> None:
    emb = FakeEmbedder(dim=64)
    store = VectorStore(emb)
    store.add("doc_add", "def add(a, b): return a + b")
    store.add("doc_sub", "def sub(a, b): return a - b")
    store.add("doc_http", "open a network socket and send bytes")

    hits = store.search("def add(a, b): return a + b", top_k=2)
    assert len(hits) == 2
    # exact-match doc should rank first
    assert hits[0].id == "doc_add"
    # scores are sorted descending
    assert hits[0].score >= hits[1].score


def test_vector_store_topk_caps_at_corpus_size() -> None:
    emb = FakeEmbedder(dim=16)
    store = VectorStore(emb)
    store.add("a", "text a")
    store.add("b", "text b")
    hits = store.search("text a", top_k=10)
    assert len(hits) == 2


def test_vector_store_empty_search() -> None:
    store = VectorStore(FakeEmbedder())
    assert store.search("anything", top_k=5) == []


def test_vector_store_add_many_and_len() -> None:
    store = VectorStore(FakeEmbedder(dim=16))
    store.add_many({"x": "text x", "y": "text y"})
    assert len(store) == 2


def test_vector_store_top_k_zero_or_negative() -> None:
    store = VectorStore(FakeEmbedder(dim=16))
    store.add("a", "alpha")
    assert store.search("alpha", top_k=0) == []
    assert store.search("alpha", top_k=-3) == []


def test_vector_store_overwrites_same_id() -> None:
    store = VectorStore(FakeEmbedder(dim=16))
    store.add("a", "original text")
    store.add("a", "replacement text")
    assert len(store) == 1


def test_vector_store_add_many_empty_is_noop() -> None:
    store = VectorStore(FakeEmbedder(dim=16))
    store.add_many({})
    assert len(store) == 0


def test_fake_embedder_rejects_non_positive_dim() -> None:
    with pytest.raises(ValueError):
        FakeEmbedder(dim=0)
    with pytest.raises(ValueError):
        FakeEmbedder(dim=-5)


# --------------------------------------------------------------------------- #
# BedrockEmbedder stub: lazy + deferred                                         #
# --------------------------------------------------------------------------- #
def test_bedrock_embedder_imports_without_boto3() -> None:
    """Constructing the stub must not require boto3 (lazy real impl)."""
    emb = BedrockEmbedder(model_id="amazon.titan-embed-text-v2:0")
    assert isinstance(emb, Embedder)
    assert emb.dim > 0


def test_bedrock_embedder_embed_is_not_implemented() -> None:
    emb = BedrockEmbedder()
    with pytest.raises(NotImplementedError):
        emb.embed(["hello"])


def test_bedrock_embedder_conforms_to_interface() -> None:
    assert isinstance(BedrockEmbedder(), Embedder)
