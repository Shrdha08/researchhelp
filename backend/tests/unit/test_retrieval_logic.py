from researchhelp.retrieval.retriever import select_balanced
from researchhelp.retrieval.vectorstore import ScoredChunk, build_filter


def hit(paper, idx, score):
    return ScoredChunk({"chunk_id": f"{paper}:{idx}", "paper_id": paper, "text": "t"}, score)


def test_select_balanced_guarantees_each_paper_a_slot():
    candidates = {
        "A": [hit("A", i, 0.9 - i * 0.01) for i in range(8)],  # A dominates on raw score
        "B": [hit("B", 0, 0.5), hit("B", 1, 0.4)],
    }
    chosen = select_balanced(candidates, k_final=4, min_per_paper=1)
    papers = [h.payload["paper_id"] for h in chosen]
    assert papers.count("B") == 1 and papers.count("A") == 3
    assert [h.score for h in chosen] == sorted((h.score for h in chosen), reverse=True)


def test_select_balanced_without_guarantee_is_pure_top_k():
    candidates = {"A": [hit("A", i, 0.9 - i * 0.01) for i in range(8)], "B": [hit("B", 0, 0.5)]}
    chosen = select_balanced(candidates, k_final=4, min_per_paper=0)
    assert {h.payload["paper_id"] for h in chosen} == {"A"}


def test_select_balanced_handles_empty_paper_and_small_pool():
    chosen = select_balanced({"A": [hit("A", 0, 0.3)], "B": []}, k_final=6)
    assert len(chosen) == 1


def test_select_balanced_no_duplicates():
    candidates = {"A": [hit("A", i, 1 - i / 10) for i in range(5)]}
    chosen = select_balanced(candidates, k_final=3, min_per_paper=2)
    assert len({h.payload["chunk_id"] for h in chosen}) == 3


def test_build_filter_scopes_papers_and_excludes_references():
    flt = build_filter(["p1", "p2"])
    assert flt.must[0].key == "paper_id" and flt.must[0].match.any == ["p1", "p2"]
    assert flt.must_not[0].key == "section" and flt.must_not[0].match.any == ["References"]


def test_build_filter_empty():
    assert build_filter(None, ()) is None
