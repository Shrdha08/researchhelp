import pytest

from researchhelp.retrieval.retriever import STRATEGIES, ScopedRetriever
from tests.fakes import KeywordReranker

pytestmark = pytest.mark.integration


def make(store, strategy, **kw):
    reranker = KeywordReranker() if strategy == "hybrid_rerank" else None
    return ScopedRetriever(store, strategy=strategy, reranker=reranker, **kw)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_strategy_respects_scope_and_k_final(indexed, strategy):
    store, widget, gadget = indexed
    docs = make(store, strategy, k_final=3).retrieve("datasets used", [gadget.paper_id])
    assert len(docs) == 3
    assert {d.metadata["paper_id"] for d in docs} == {gadget.paper_id}
    assert [d.metadata["rank"] for d in docs] == [1, 2, 3]
    assert all(d.metadata["section"] != "References" for d in docs)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_strategy_fans_out_across_papers(indexed, strategy):
    store, widget, gadget = indexed
    docs = make(store, strategy, k_final=3).retrieve(
        "molecular graphs message passing molecules QM9", [widget.paper_id, gadget.paper_id]
    )
    assert {d.metadata["paper_id"] for d in docs} == {widget.paper_id, gadget.paper_id}


def test_hybrid_finds_exact_term(indexed):
    store, widget, _ = indexed
    docs = make(store, "hybrid", k_final=2).retrieve("Covertype", [widget.paper_id])
    assert "Covertype" in docs[0].page_content


def test_rerank_pool_size_and_reordering(indexed):
    store, widget, gadget = indexed
    retriever = make(store, "hybrid_rerank", k_final=2, k_per_paper=3, k_candidates=10)
    docs = retriever.retrieve("limitations 3D conformations small molecules", [gadget.paper_id])
    # One paper: the pool is k_candidates (bounded by the paper's chunks), not k_per_paper.
    assert retriever.reranker.calls[0] == min(10, store.count(gadget.paper_id))
    assert docs[0].metadata["section"] == "Limitations"
    assert docs[0].metadata["score"] >= docs[1].metadata["score"]


def test_rerank_pool_is_split_across_papers(indexed):
    store, widget, gadget = indexed
    retriever = make(store, "hybrid_rerank", k_final=4, k_per_paper=2, k_candidates=6)
    retriever.retrieve("datasets", [widget.paper_id, gadget.paper_id])
    assert retriever.reranker.calls[0] <= 6


def test_rerank_requires_reranker(store):
    with pytest.raises(ValueError):
        ScopedRetriever(store, strategy="hybrid_rerank")
