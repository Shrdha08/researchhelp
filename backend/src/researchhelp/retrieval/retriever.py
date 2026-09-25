"""ScopedRetriever: the single retrieval entry point used by every RAG pipeline.

Scope = the list of paper IDs selected by the user. A single paper is a list of length one;
there is no separate single-paper / multi-paper code path.

For several papers we *fan out*: retrieve top-k for each paper separately (filter
``paper_id == X``), then select the final chunks with a per-paper minimum. Without fan-out, one
long or very on-topic paper can fill every slot and a comparison would silently ignore the
other papers.
"""

from collections.abc import Sequence
from typing import Literal

from langchain_core.documents import Document

from researchhelp.retrieval.vectorstore import PaperVectorStore, ScoredChunk

Strategy = Literal["semantic"]  # "hybrid" and "hybrid_rerank" are added in Phase 2


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
    ):
        if strategy != "semantic":
            raise ValueError(f"Unsupported retrieval strategy: {strategy!r}")
        self.store = store
        self.k_per_paper = k_per_paper
        self.k_final = k_final
        self.min_per_paper = min_per_paper
        self.strategy = strategy

    def retrieve(self, query: str, paper_ids: Sequence[str]) -> list[Document]:
        if not paper_ids:
            raise ValueError("Select at least one paper to search.")
        paper_ids = list(dict.fromkeys(paper_ids))  # de-duplicate, keep order

        query_vector = self.store.dense.embed_query(query)  # embed once, reuse per paper
        candidates = {
            pid: self.store.search_dense(query_vector, [pid], limit=self.k_per_paper)
            for pid in paper_ids
        }
        selected = select_balanced(candidates, self.k_final, self.min_per_paper)
        return [to_document(hit, rank) for rank, hit in enumerate(selected, start=1)]
