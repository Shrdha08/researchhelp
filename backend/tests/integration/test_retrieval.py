import pytest

from researchhelp.ingestion.pipeline import ingest_pdf
from researchhelp.retrieval.retriever import ScopedRetriever

pytestmark = pytest.mark.integration


def test_ingest_stores_every_chunk_with_metadata(indexed):
    store, widget, gadget = indexed
    assert store.count(widget.paper_id) == len(widget.chunks)
    assert store.count(gadget.paper_id) == len(gadget.chunks)
    papers = {p.paper_id: p for p in store.list_papers()}
    assert papers[widget.paper_id].title == "Sparse Widget Networks for Tabular Data"
    assert papers[widget.paper_id].num_pages == 3


def test_reingest_is_idempotent(indexed, widget_pdf):
    store, widget, _ = indexed
    ingest_pdf(widget_pdf, store, chunk_size=300, chunk_overlap=50)
    assert store.count(widget.paper_id) == len(widget.chunks)


def test_single_paper_scope_only_returns_that_paper(indexed):
    store, widget, gadget = indexed
    retriever = ScopedRetriever(store, k_per_paper=5, k_final=4)
    docs = retriever.retrieve("which dataset and optimizer were used", [gadget.paper_id])
    assert docs and {d.metadata["paper_id"] for d in docs} == {gadget.paper_id}


def test_single_paper_retrieval_finds_the_right_page_and_section(indexed):
    store, widget, _ = indexed
    docs = ScopedRetriever(store, k_final=3).retrieve(
        "AdamW optimizer learning rate epochs Adult dataset", [widget.paper_id]
    )
    top = docs[0]
    assert "AdamW" in top.page_content
    assert (top.metadata["page"], top.metadata["section"]) == (3, "Experiments")
    assert top.metadata["rank"] == 1 and "score" in top.metadata


def test_references_are_excluded(indexed):
    store, widget, _ = indexed
    docs = ScopedRetriever(store, k_per_paper=20, k_final=20).retrieve(
        "XGBoost scalable tree boosting system Chen Guestrin KDD", [widget.paper_id]
    )
    assert all(d.metadata["section"] != "References" for d in docs)


def test_multi_paper_fan_out_includes_every_selected_paper(indexed):
    store, widget, gadget = indexed
    # The query is about molecules, so without fan-out the widget paper would lose every slot.
    docs = ScopedRetriever(store, k_per_paper=5, k_final=3).retrieve(
        "molecular graphs message passing molecules QM9", [widget.paper_id, gadget.paper_id]
    )
    assert {d.metadata["paper_id"] for d in docs} == {widget.paper_id, gadget.paper_id}


def test_delete_paper_removes_only_its_points(indexed):
    store, widget, gadget = indexed
    store.delete_paper(widget.paper_id)
    assert store.count(widget.paper_id) == 0
    assert store.count(gadget.paper_id) == len(gadget.chunks)


def test_retrieve_requires_scope(indexed):
    store, *_ = indexed
    with pytest.raises(ValueError):
        ScopedRetriever(store).retrieve("anything", [])


def test_without_per_paper_guarantee_one_paper_can_take_every_slot(indexed):
    """Counterfactual for the fan-out test above: this is the failure mode fan-out prevents."""
    store, widget, gadget = indexed
    docs = ScopedRetriever(store, k_per_paper=5, k_final=3, min_per_paper=0).retrieve(
        "molecular graphs message passing molecules QM9", [widget.paper_id, gadget.paper_id]
    )
    assert {d.metadata["paper_id"] for d in docs} == {gadget.paper_id}
