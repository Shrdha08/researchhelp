"""Dense and sparse text encoders.

Two small protocols decouple retrieval from any specific model library: production uses
FastEmbed (ONNX, CPU), tests use cheap deterministic encoders.
"""

from collections.abc import Sequence
from typing import Protocol

from qdrant_client import models

from researchhelp.retrieval.onnx_device import log_provider, onnx_providers

# BGE retrieval models are trained with this instruction on the *query* side only; passages
# are embedded as-is. FastEmbed's query_embed does not add it, so we do.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class DenseEncoder(Protocol):
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SparseEncoder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[models.SparseVector]: ...

    def embed_query(self, text: str) -> models.SparseVector: ...


class FastEmbedDense:
    def __init__(self, model_name: str, batch_size: int = 32, device: str = "auto"):
        from fastembed import TextEmbedding

        providers = onnx_providers(device)
        self.model_name = model_name
        self._model = TextEmbedding(model_name, providers=providers)
        self.provider = log_provider(model_name, self._model, providers)
        self.dim = self._model.embedding_size
        self._batch_size = batch_size
        self._query_prefix = BGE_QUERY_INSTRUCTION if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(list(texts), batch_size=self._batch_size)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.embed([self._query_prefix + text]))).tolist()


class FastEmbedSparse:
    """BM25 term weights. IDF is applied by Qdrant at query time (``Modifier.IDF``), so the
    statistics stay correct as papers are added or deleted."""

    def __init__(self, model_name: str = "Qdrant/bm25"):
        from fastembed import SparseTextEmbedding

        self.model_name = model_name
        self._model = SparseTextEmbedding(model_name)

    @staticmethod
    def _to_qdrant(emb) -> models.SparseVector:
        return models.SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())

    def embed_documents(self, texts: Sequence[str]) -> list[models.SparseVector]:
        return [self._to_qdrant(e) for e in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> models.SparseVector:
        return self._to_qdrant(next(iter(self._model.query_embed(text))))
