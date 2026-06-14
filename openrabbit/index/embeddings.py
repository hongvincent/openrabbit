"""Offline embeddings interface + tiny in-memory vector store (item 16).

The vector path is **interface-ready but offline-safe**. The default
:class:`FakeEmbedder` is deterministic and hashing-based (no network, no native
deps), so unit tests and the default pipeline never call out. A
:class:`BedrockEmbedder` stub exists so the real (Titan/Cohere-on-Bedrock) path
can slot in later behind the same interface — it lazily imports boto3 and, for
v1, raises :class:`NotImplementedError` rather than making a network call.

Storage is kept behind the :class:`VectorStore` interface (cosine top-k in
memory) so a durable backend (LanceDB-on-S3 / pgvector-on-RDS, SPEC §15) can be
swapped in later without touching callers. That richer backend is intentionally
documented but NOT implemented here.
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

Vector = list[float]

# Default embedding dimension for the hashing FakeEmbedder. Small but enough to
# separate distinct strings deterministically for top-k tests/demos.
_DEFAULT_DIM = 256


class Embedder(ABC):
    """Interface for turning text into fixed-length vectors.

    Implementations MUST be deterministic per ``(text, dim)`` for the vector
    store's results to be reproducible. Cloud-backed implementations import
    their SDK lazily so this module imports with zero external dependencies.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the vectors this embedder produces."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[Vector]:
        """Embed a batch of texts into unit-length vectors (one per text)."""


class FakeEmbedder(Embedder):
    """Deterministic, hashing-based offline embedder (no network).

    Each text is hashed into a fixed-dimension vector by seeding a stream of
    SHA-256 digests and unpacking bytes into signed floats, then L2-normalized.
    Identical inputs always produce identical vectors; distinct inputs almost
    always produce distinct vectors. Purely for offline testing/demos and as the
    default so the vector path is exercisable without credentials.
    """

    def __init__(self, dim: int = _DEFAULT_DIM) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[Vector]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> Vector:
        # Generate enough deterministic bytes by chaining counter-salted digests.
        raw = bytearray()
        counter = 0
        seed = text.encode("utf-8")
        while len(raw) < self._dim:
            digest = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
            raw.extend(digest)
            counter += 1
        # Map each byte (0..255) to a signed float centred at 0.
        vec = [(raw[i] - 127.5) / 127.5 for i in range(self._dim)]
        return _normalize(vec)


class BedrockEmbedder(Embedder):
    """Lazy stub for Bedrock text embeddings (real impl DEFERRED, SPEC §15).

    Constructing the stub must not require ``boto3``; the SDK is imported lazily
    only when a real call is made. For v1 :meth:`embed` raises
    :class:`NotImplementedError` so the interface is complete and selectable
    without pulling the embeddings pipeline into the default (agentic-first)
    flow.
    """

    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v2:0",
        *,
        region: str = "us-east-1",
        dim: int = 1024,
    ) -> None:
        self.model_id = model_id
        self.region = region
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[Vector]:  # pragma: no cover - stub
        # Real impl (deferred): lazily ``import boto3``, call
        # bedrock-runtime InvokeModel for self.model_id, parse the embedding.
        raise NotImplementedError(
            "BedrockEmbedder is a v1 stub; real Bedrock embeddings are deferred "
            "(SPEC §15). Use FakeEmbedder for offline tests/demos."
        )


@dataclass(frozen=True)
class SearchHit:
    """One vector-store search result."""

    id: str
    score: float
    text: str


class VectorStore:
    """A tiny in-memory vector store with cosine top-k retrieval.

    Storage lives behind this class deliberately: a durable backend
    (LanceDB-on-S3 or pgvector-on-RDS, SPEC §15) can replace the internal dict
    without changing the ``add`` / ``search`` surface that callers use. That
    richer backend is documented but NOT implemented in v1.
    """

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        # id -> (vector, text)
        self._items: dict[str, tuple[Vector, str]] = {}

    def __len__(self) -> int:
        return len(self._items)

    def add(self, doc_id: str, text: str) -> None:
        """Embed and store ``text`` under ``doc_id`` (overwrites same id)."""
        [vec] = self._embedder.embed([text])
        self._items[doc_id] = (vec, text)

    def add_many(self, docs: Mapping[str, str]) -> None:
        """Embed and store many ``{id: text}`` documents in one batch."""
        ids = list(docs.keys())
        if not ids:
            return
        vectors = self._embedder.embed([docs[i] for i in ids])
        for doc_id, vec in zip(ids, vectors):
            self._items[doc_id] = (vec, docs[doc_id])

    def search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        """Return the ``top_k`` documents most cosine-similar to ``query``."""
        if top_k <= 0 or not self._items:
            return []
        [q] = self._embedder.embed([query])
        scored = [
            SearchHit(id=doc_id, score=cosine_similarity(q, vec), text=text)
            for doc_id, (vec, text) in self._items.items()
        ]
        # Sort by score desc, then id for stable deterministic ordering.
        scored.sort(key=lambda h: (-h.score, h.id))
        return scored[:top_k]


# --------------------------------------------------------------------------- #
# math helpers                                                                  #
# --------------------------------------------------------------------------- #
def cosine_similarity(a: Vector, b: Vector) -> float:
    """Cosine similarity of two equal-length vectors (0.0 for a zero vector)."""
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} != {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _normalize(vec: Vector) -> Vector:
    """L2-normalize a vector; an all-zero vector is returned unchanged."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]
