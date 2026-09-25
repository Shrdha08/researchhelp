"""Builds retrieval components from Settings (one place that knows about concrete models)."""

from functools import lru_cache

from qdrant_client import QdrantClient

from researchhelp.config import Settings, get_settings
from researchhelp.retrieval.embeddings import FastEmbedDense, FastEmbedSparse
from researchhelp.retrieval.retriever import ScopedRetriever
from researchhelp.retrieval.vectorstore import PaperVectorStore


def make_qdrant_client(settings: Settings) -> QdrantClient:
    if settings.qdrant_path:
        settings.qdrant_path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(settings.qdrant_path))
    return QdrantClient(url=settings.qdrant_url)


@lru_cache
def get_store() -> PaperVectorStore:
    settings = get_settings()
    store = PaperVectorStore(
        client=make_qdrant_client(settings),
        dense=FastEmbedDense(settings.embed_model),
        sparse=FastEmbedSparse(settings.sparse_model),
        collection=settings.qdrant_collection,
    )
    store.ensure_collection()
    return store


def get_retriever() -> ScopedRetriever:
    settings = get_settings()
    return ScopedRetriever(get_store(), k_per_paper=settings.k_per_paper, k_final=settings.k_final)
