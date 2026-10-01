from langchain_core.documents import Document

from researchhelp.rag.common.context import format_context, resolve_citations


def doc(paper="p1", page=3, section="Experiments", text="We use AdamW."):
    return Document(
        page_content=text,
        metadata={
            "paper_id": paper,
            "paper_title": f"Title {paper}",
            "page": page,
            "section": section,
            "chunk_id": f"{paper}:p{page}:c0",
        },
    )


def test_format_context_numbers_sources_and_shows_metadata():
    context, sources = format_context([doc(), doc("p2", 7, "Method", "We use SGD.")])
    assert list(sources) == ["S1", "S2"]
    assert "[S1] (Paper: Title p1 | Page 3 | Section: Experiments)\nWe use AdamW." in context
    assert "[S2] (Paper: Title p2 | Page 7 | Section: Method)" in context


def test_resolve_citations_maps_ids_to_pages_in_order_of_use():
    _, sources = format_context([doc(), doc("p2", 7)])
    text, cites = resolve_citations("B uses SGD [S2]. A uses AdamW [S1, S2].", sources)
    assert text == "B uses SGD [S2]. A uses AdamW [S1, S2]."
    assert [(c.source_id, c.paper_id, c.page) for c in cites] == [("S2", "p2", 7), ("S1", "p1", 3)]


def test_resolve_citations_drops_ids_not_in_context():
    _, sources = format_context([doc()])
    text, cites = resolve_citations("AdamW [S1]. Batch size 128 [S9]. Both [S1; S7].", sources)
    assert text == "AdamW [S1]. Batch size 128. Both [S1]."
    assert [c.source_id for c in cites] == ["S1"]


def test_resolve_citations_without_markers():
    _, sources = format_context([doc()])
    assert resolve_citations("Not stated in the excerpts.", sources) == (
        "Not stated in the excerpts.",
        [],
    )


def test_snippet_is_truncated():
    _, sources = format_context([doc(text="word " * 200)])
    _, (cite,) = resolve_citations("x [S1]", sources)
    assert len(cite.snippet) <= 305 and cite.snippet.endswith("...")
