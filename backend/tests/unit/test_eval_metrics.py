"""Metric arithmetic of evaluation/retrieval_eval.py (wrong metrics silently invalidate results)."""

import sys
from pathlib import Path

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root, for `evaluation`

from evaluation.common import Evidence, Question, chunk_matches, split_for  # noqa: E402
from evaluation.retrieval_eval import score_question  # noqa: E402


def doc(pid, page, text):
    return Document(page_content=text, metadata={"paper_id": pid, "page": page})


def question(evidence):
    return Question("q", "comparison", ["a", "b"], ["A", "B"], "?", "", evidence)


EV_A = Evidence("a", "A", 3, "batch size of 128")
EV_B = Evidence("b", "B", 5, "learning rate of 1e-4")


def test_chunk_match_needs_paper_page_and_quote():
    assert chunk_matches(EV_A, "A", 3, "We use a Batch  size of 128 here.")  # case, spacing
    assert not chunk_matches(EV_A, "A", 4, "batch size of 128")  # wrong page
    assert not chunk_matches(EV_A, "B", 3, "batch size of 128")  # wrong paper
    assert not chunk_matches(EV_A, "A", 3, "batch size of 64")  # quote missing


def test_recall_counts_evidence_items_and_mrr_uses_first_hit():
    docs = [doc("A", 1, "x")] * 2 + [doc("A", 3, "batch size of 128")] + [doc("A", 2, "y")] * 3
    docs += [doc("B", 5, "learning rate of 1e-4")]  # rank 7
    s = score_question(question([EV_A, EV_B]), docs)
    assert s["recall@5"] == 0.5  # only paper A's evidence in the top 5
    assert s["recall@10"] == 1.0
    assert s["mrr"] == 1 / 3
    assert s["first_relevant_rank"] == 3


def test_no_hit_gives_zero():
    s = score_question(question([EV_A]), [doc("A", 3, "unrelated")])
    assert s["recall@5"] == 0 and s["mrr"] == 0 and s["page_recall@5"] == 1.0


def test_split_is_deterministic():
    assert split_for("dpr-01") == split_for("dpr-01") and split_for("x") in {"dev", "test"}
