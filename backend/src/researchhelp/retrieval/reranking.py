"""Cross-encoder reranking.

First-stage retrieval uses a *bi-encoder*: query and chunk are embedded independently, which is
fast (chunk vectors are precomputed) but the two texts never "see" each other. A
*cross-encoder* reads the query and the chunk together in one forward pass and outputs a
relevance score. It is far more accurate, but too slow to run over the whole collection, so it
only re-orders a small candidate pool (~20 chunks) produced by the first stage.
"""

from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

from researchhelp.retrieval.vectorstore import ScoredChunk


class Reranker(Protocol):
    def rerank(self, query: str, hits: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        """Return ``hits`` re-scored by relevance to ``query``, best first."""
        ...


def _apply_scores(hits: Sequence[ScoredChunk], scores: Sequence[float]) -> list[ScoredChunk]:
    rescored = [replace(h, score=float(s)) for h, s in zip(hits, scores, strict=True)]
    return sorted(rescored, key=lambda h: h.score, reverse=True)


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", batch_size: int = 16):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name
        self._model = TextCrossEncoder(model_name)
        self._batch_size = batch_size

    def rerank(self, query: str, hits: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        if not hits:
            return []
        texts = [h.payload["text"] for h in hits]
        scores = list(self._model.rerank(query, texts, batch_size=self._batch_size))
        return _apply_scores(hits, scores)
