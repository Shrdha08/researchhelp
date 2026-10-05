"""Builds retrieval components from Settings (one place that knows about concrete models)."""

import atexit
from functools import lru_cache

from qdrant_client import QdrantClient

from researchhelp.config import Settings, get_settings
from researchhelp.retrieval.embeddings import FastEmbedDense, FastEmbedSparse
from researchhelp.retrieval.reranking import CrossEncoderReranker
from researchhelp.retrieval.retriever import ScopedRetriever
from researchhelp.retrieval.vectorstore import PaperVectorStore


def make_qdrant_client(settings: Settings) -> QdrantClient:
    if settings.qdrant_path:
        settings.qdrant_path.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(settings.qdrant_path))
    else:
        client = QdrantClient(url=settings.qdrant_url)
    # Close before interpreter shutdown; local mode otherwise errors in __del__.
    atexit.register(client.close)
    return client


@lru_cache
def get_store() -> PaperVectorStore:
    settings = get_settings()
    store = PaperVectorStore(
        client=make_qdrant_client(settings),
        dense=FastEmbedDense(settings.embed_model, device=settings.onnx_device),
        sparse=FastEmbedSparse(settings.sparse_model),
        collection=settings.qdrant_collection,
    )
    store.ensure_collection()
    return store


@lru_cache
def get_reranker() -> CrossEncoderReranker:
    settings = get_settings()
    return CrossEncoderReranker(settings.rerank_model, device=settings.onnx_device)


def get_retriever(strategy: str | None = None, **overrides) -> ScopedRetriever:
    """Retriever configured from Settings; ``overrides`` (k_final, k_candidates, ...) are used by
    evaluation sweeps so that experiments exercise exactly the production code path."""
    settings = get_settings()
    strategy = strategy or settings.retrieval_strategy
    params = {
        "k_per_paper": settings.k_per_paper,
        "k_final": settings.k_final,
        "k_candidates": settings.k_candidates,
        "min_per_paper": settings.min_per_paper,
        **overrides,
    }
    reranker = get_reranker() if strategy == "hybrid_rerank" else None
    return ScopedRetriever(get_store(), strategy=strategy, reranker=reranker, **params)
