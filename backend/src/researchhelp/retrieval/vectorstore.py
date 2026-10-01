"""Qdrant access: collection schema, writes, scoped searches, deletes.

One collection holds every chunk of every paper. Each point has a named dense vector and a
named sparse (BM25) vector; the paper scope is expressed as a payload filter on ``paper_id``.
"""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from researchhelp.ingestion.chunking import Chunk
from researchhelp.ingestion.metadata.sections import REFERENCES
from researchhelp.retrieval.embeddings import DenseEncoder, SparseEncoder

DENSE = "dense"
SPARSE = "sparse"
DEFAULT_EXCLUDED_SECTIONS = (REFERENCES,)


@dataclass(frozen=True)
class ScoredChunk:
    payload: dict
    score: float


@dataclass(frozen=True)
class PaperSummary:
    paper_id: str
    title: str
    num_chunks: int
    num_pages: int


def build_filter(
    paper_ids: Sequence[str] | None = None,
    exclude_sections: Sequence[str] = DEFAULT_EXCLUDED_SECTIONS,
) -> models.Filter | None:
    must = []
    if paper_ids:
        must.append(
            models.FieldCondition(key="paper_id", match=models.MatchAny(any=list(paper_ids)))
        )
    must_not = []
    if exclude_sections:
        must_not.append(
            models.FieldCondition(key="section", match=models.MatchAny(any=list(exclude_sections)))
        )
    if not must and not must_not:
        return None
    return models.Filter(must=must or None, must_not=must_not or None)


class PaperVectorStore:
    def __init__(
        self,
        client: QdrantClient,
        dense: DenseEncoder,
        sparse: SparseEncoder,
        collection: str = "paper_chunks",
    ):
        self.client = client
        self.dense = dense
        self.sparse = sparse
        self.collection = collection

    # ----- schema -----------------------------------------------------------------------------

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            self.collection,
            vectors_config={
                DENSE: models.VectorParams(size=self.dense.dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        with warnings.catch_warnings():
            # Embedded (local) Qdrant ignores payload indexes and warns; the server uses them.
            warnings.filterwarnings("ignore", message="Payload indexes have no effect")
            for field in ("paper_id", "section"):
                self.client.create_payload_index(
                    self.collection, field, models.PayloadSchemaType.KEYWORD
                )

    # ----- writes -----------------------------------------------------------------------------

    def upsert_chunks(self, chunks: Sequence[Chunk], batch_size: int = 64) -> int:
        """Embed and store chunks. Point IDs are UUIDv5(chunk_id), so re-ingesting the same
        paper overwrites its points instead of duplicating them."""
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [c.text for c in batch]
            dense_vecs = self.dense.embed_documents(texts)
            sparse_vecs = self.sparse.embed_documents(texts)
            points = [
                models.PointStruct(
                    id=chunk.point_id,
                    vector={DENSE: d, SPARSE: s},
                    payload=chunk.payload(),
                )
                for chunk, d, s in zip(batch, dense_vecs, sparse_vecs, strict=True)
            ]
            self.client.upsert(self.collection, points=points, wait=True)
        return len(chunks)

    def delete_paper(self, paper_id: str) -> None:
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(filter=build_filter([paper_id], ())),
            wait=True,
        )

    # ----- reads ------------------------------------------------------------------------------

    def count(self, paper_id: str | None = None) -> int:
        flt = build_filter([paper_id], ()) if paper_id else None
        return self.client.count(self.collection, count_filter=flt, exact=True).count

    def search_dense(
        self,
        query_vector: list[float],
        paper_ids: Sequence[str],
        limit: int,
        exclude_sections: Sequence[str] = DEFAULT_EXCLUDED_SECTIONS,
    ) -> list[ScoredChunk]:
        result = self.client.query_points(
            self.collection,
            query=query_vector,
            using=DENSE,
            query_filter=build_filter(paper_ids, exclude_sections),
            limit=limit,
            with_payload=True,
        )
        return [ScoredChunk(p.payload, p.score) for p in result.points]

    def list_papers(self) -> list[PaperSummary]:
        """Distinct papers in the collection. Scans payloads, which is fine for a personal
        library of tens of papers; PostgreSQL becomes the paper registry in Phase 5."""
        stats: dict[str, dict] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection,
                limit=256,
                offset=offset,
                with_payload=["paper_id", "paper_title", "page"],
                with_vectors=False,
            )
            for p in points:
                s = stats.setdefault(
                    p.payload["paper_id"],
                    {"title": p.payload["paper_title"], "chunks": 0, "pages": 0},
                )
                s["chunks"] += 1
                s["pages"] = max(s["pages"], p.payload["page"])
            if offset is None:
                break
        return [PaperSummary(pid, s["title"], s["chunks"], s["pages"]) for pid, s in stats.items()]
