from langchain_core.documents import Document

from researchhelp.rag.common.context import format_context
from researchhelp.rag.research import prompts
from researchhelp.rag.research.chain import merge_ranked, validate_evidence, validate_synthesis
from researchhelp.rag.research.schemas import (
    AnalysisItem,
    DirectionItem,
    EvidenceExtraction,
    ExtractedEvidence,
    Synthesis,
    normalize_ids,
)


def doc(cid, page=1, title="Paper A"):
    return Document(
        page_content=f"text {cid}",
        metadata={
            "chunk_id": cid,
            "paper_id": "pa",
            "paper_title": title,
            "page": page,
            "section": "Method",
        },
    )


def test_normalize_ids_accepts_model_variants():
    assert normalize_ids(["[S1]", "【S2】", "s 3", "4", "S1", "E5"], "S") == [
        "S1",
        "S2",
        "S3",
        "S4",
    ]
    assert normalize_ids("E1, e2; E1", "E") == ["E1", "E2"]


def test_merge_ranked_interleaves_and_dedupes():
    a = [doc("c1"), doc("c2"), doc("c3")]
    b = [doc("c2"), doc("c9")]
    merged = merge_ranked([a, b], limit=10)
    # rank 0: c1, c2; rank 1: (c2 duplicate), c9; rank 2: c3
    assert [d.metadata["chunk_id"] for d in merged] == ["c1", "c2", "c9", "c3"]
    assert len(merge_ranked([a, b], limit=2)) == 2


def test_validate_evidence_drops_unsupported_claims_and_numbers_the_rest():
    _, sources = format_context([doc("c1", page=3), doc("c2", page=7, title="Paper B")])
    extraction = EvidenceExtraction(
        evidence=[
            ExtractedEvidence(claim="Uses BM25.", sources=["S9"]),  # unknown source
            ExtractedEvidence(claim="Trained on NQ.", sources=["[S2]"]),
            ExtractedEvidence(claim="   ", sources=["S1"]),  # empty claim
            ExtractedEvidence(claim="Assumes static corpus.", sources=["S1", "S7"]),
        ]
    )
    items = validate_evidence(extraction, sources)
    assert [(e.id, e.claim, e.paper_title) for e in items] == [
        ("E1", "Trained on NQ.", "Paper B"),
        ("E2", "Assumes static corpus.", "Paper A"),
    ]
    assert [c.page for c in items[1].citations] == [3]  # S7 silently dropped


def test_validate_synthesis_requires_real_evidence_links():
    synthesis = Synthesis(
        analysis=[
            AnalysisItem(statement="ok", based_on=["e1"]),
            AnalysisItem(statement="ungrounded", based_on=[]),
            AnalysisItem(statement="bad ref", based_on=["E9"]),
        ],
        directions=[DirectionItem(title="t", based_on=["E2", "E9"])],
    )
    kept = validate_synthesis(synthesis, {"E1", "E2"})
    assert [a.statement for a in kept.analysis] == ["ok"]
    assert kept.analysis[0].based_on == ["E1"]
    assert kept.directions[0].based_on == ["E2"]


def test_confidence_is_normalised():
    assert AnalysisItem(statement="s", confidence="HIGH").confidence == "high"
    assert AnalysisItem(statement="s", confidence="very").confidence == "medium"


def test_prompts_separate_evidence_from_inference():
    assert "interpretation" in prompts.EVIDENCE_SYSTEM
    assert "hypotheses" in prompts.SYNTHESIS_SYSTEM
    assert "do not see the papers" in prompts.SYNTHESIS_SYSTEM
    assert "inferences" in prompts.SYNTHESIS_SYSTEM
