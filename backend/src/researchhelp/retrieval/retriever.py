"""ScopedRetriever: the single retrieval entry point used by every RAG pipeline.

Scope = the list of paper IDs selected by the user. A single paper is a list of length one;
there is no separate single-paper / multi-paper code path.

For several papers we *fan out*: retrieve top-k for each paper separately (filter
``paper_id == X``), then select the final chunks with a per-paper minimum. Without fan-out, one
long or very on-topic paper can fill every slot and a comparison would silently ignore the
other papers.

Strategies (compared in evaluation/; all return the same number of chunks, ``k_final``):
  semantic       dense vector search only (Phase 1 baseline)
  hybrid         dense + BM25 sparse, fused with RRF inside Qdrant
  hybrid_rerank  hybrid candidates (~k_candidates in total) re-scored by a cross-encoder
"""

from collections.abc import Sequence
from typing import Literal

from langchain_core.documents import Document

from researchhelp.retrieval.reranking import Reranker
from researchhelp.retrieval.vectorstore import PaperVectorStore, ScoredChunk

Strategy = Literal["semantic", "hybrid", "hybrid_rerank"]
STRATEGIES: tuple[Strategy, ...] = ("semantic", "hybrid", "hybrid_rerank")


def select_balanced(
    candidates: dict[str, list[ScoredChunk]], k_final: int, min_per_paper: int = 1
) -> list[ScoredChunk]:
    """Pick ``k_final`` chunks: first the top ``min_per_paper`` of every paper (so each selected
    paper is represented when it has any match), then fill up by global score.
    ``candidates`` maps paper_id -> that paper's hits sorted by descending score."""
    chosen: list[ScoredChunk] = []
    taken: set[str] = set()

    # Papers whose best hit is strongest get their guaranteed slots first.
    by_best = sorted(
        candidates.items(), key=lambda kv: kv[1][0].score if kv[1] else 0, reverse=True
    )
    for _, hits in by_best:
        for hit in hits[:min_per_paper]:
            if len(chosen) < k_final:
                chosen.append(hit)
                taken.add(hit.payload["chunk_id"])

    rest = sorted(
        (h for hits in candidates.values() for h in hits if h.payload["chunk_id"] not in taken),
        key=lambda h: h.score,
        reverse=True,
    )
    chosen.extend(rest[: k_final - len(chosen)])
    return sorted(chosen, key=lambda h: h.score, reverse=True)


def to_document(hit: ScoredChunk, rank: int) -> Document:
    metadata = {k: v for k, v in hit.payload.items() if k != "text"}
    metadata.update(score=hit.score, rank=rank)
    return Document(page_content=hit.payload["text"], metadata=metadata)


class ScopedRetriever:
    def __init__(
        self,
        store: PaperVectorStore,
        k_per_paper: int = 8,
        k_final: int = 6,
        min_per_paper: int = 1,
        strategy: Strategy = "semantic",
        k_candidates: int = 20,
        reranker: Reranker | None = None,
    ):
        if strategy not in STRATEGIES:
            raise ValueError(f"Unsupported retrieval strategy: {strategy!r}")
        if strategy == "hybrid_rerank" and reranker is None:
            raise ValueError("Strategy 'hybrid_rerank' needs a reranker.")
        self.store = store
        self.k_per_paper = k_per_paper
        self.k_final = k_final
        self.min_per_paper = min_per_paper
        self.strategy = strategy
        self.k_candidates = k_candidates
        self.reranker = reranker

    def _per_paper_limit(self, n_papers: int) -> int:
        if self.strategy != "hybrid_rerank":
            return self.k_per_paper
        # Reranking pool of about k_candidates in total, but never fewer than k_per_paper each.
        return max(self.k_per_paper, -(-self.k_candidates // n_papers))

    def _search(self, query: str, paper_ids: list[str]) -> dict[str, list[ScoredChunk]]:
        dense = self.store.dense.embed_query(query)  # embed once, reuse for every paper
        limit = self._per_paper_limit(len(paper_ids))
        if self.strategy == "semantic":
            return {pid: self.store.search_dense(dense, [pid], limit=limit) for pid in paper_ids}
        sparse = self.store.sparse.embed_query(query)
        return {
            pid: self.store.search_hybrid(
                dense, sparse, [pid], limit=limit, prefetch_limit=max(limit, self.k_candidates)
            )
            for pid in paper_ids
        }

    def retrieve(self, query: str, paper_ids: Sequence[str]) -> list[Document]:
        if not paper_ids:
            raise ValueError("Select at least one paper to search.")
        paper_ids = list(dict.fromkeys(paper_ids))  # de-duplicate, keep order

        candidates = self._search(query, paper_ids)
        if self.strategy == "hybrid_rerank":
            pool = [hit for hits in candidates.values() for hit in hits]
            reranked = self.reranker.rerank(query, pool)
            # Cross-encoder scores share one scale across papers, so regroup and rebalance.
            candidates = {pid: [] for pid in paper_ids}
            for hit in reranked:
                candidates[hit.payload["paper_id"]].append(hit)

        selected = select_balanced(candidates, self.k_final, self.min_per_paper)
        return [to_document(hit, rank) for rank, hit in enumerate(selected, start=1)]
