"""Offline-safe codebase index (SPEC 1.2 / Phase 4, item 16).

A bounded, dependency-light Greptile-style index behind a clean interface:

* :mod:`openrabbit.index.symbols` — a :class:`SymbolIndex` built with the
  standard-library :mod:`ast` (Python) + lightweight regex heuristics (js/ts/go)
  with NO heavy native deps. tree-sitter is an OPTIONAL lazily-imported backend
  that is never required. The indexer NEVER imports/execs the target repo code.
* :mod:`openrabbit.index.embeddings` — an :class:`Embedder` interface with a
  deterministic offline :class:`FakeEmbedder` default + a lazy
  :class:`BedrockEmbedder` stub, and a tiny cosine top-k :class:`VectorStore`.

The cross-file impact query :meth:`SymbolIndex.impacted_by` is exposed so the
agentic-escalation path COULD use it, but it is intentionally NOT wired into the
default Phase-0 agentic-first pipeline.
"""

from __future__ import annotations

from openrabbit.index.embeddings import (
    BedrockEmbedder,
    Embedder,
    FakeEmbedder,
    SearchHit,
    VectorStore,
    cosine_similarity,
)
from openrabbit.index.symbols import (
    Edge,
    EdgeKind,
    Node,
    NodeKind,
    SymbolIndex,
    build_index,
)

__all__ = [  # noqa: RUF022 — grouped by source module (symbols/embeddings), not alphabetical
    # symbols
    "SymbolIndex",
    "build_index",
    "Node",
    "Edge",
    "NodeKind",
    "EdgeKind",
    # embeddings
    "Embedder",
    "FakeEmbedder",
    "BedrockEmbedder",
    "VectorStore",
    "SearchHit",
    "cosine_similarity",
]
