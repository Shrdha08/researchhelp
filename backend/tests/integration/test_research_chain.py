"""Research pipeline end to end on the synthetic PDFs with scripted fake LLM output."""

import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from researchhelp.rag.research.chain import build_research_chain
from researchhelp.retrieval.retriever import ScopedRetriever

pytestmark = pytest.mark.integration
AUX = "limitations, weaknesses, assumptions, future work"


class RecordingFakeLLM(FakeListChatModel):
    prompts: list = []

    def _call(self, messages, *args, **kwargs):
        self.prompts.append(messages)
        return super()._call(messages, *args, **kwargs)


def evidence_json(*items):
    return json.dumps({"evidence": [{"claim": c, "paper": "", "sources": s} for c, s in items]})


SYNTHESIS = json.dumps(
    {
        "analysis": [
            {
                "statement": "Both models are only tested on small data.",
                "based_on": ["E1", "E2"],
                "confidence": "medium",
            },
            {"statement": "Ungrounded speculation.", "based_on": []},
        ],
        "directions": [
            {
                "title": "3D-aware gadgets",
                "rationale": "E2 notes no 3D.",
                "based_on": ["E2"],
                "validation_experiment": "Compare on QM9 with conformers.",
            }
        ],
    }
)


def test_research_chain_multi_paper(indexed):
    store, widget, gadget = indexed
    llm = RecordingFakeLLM(
        responses=[
            evidence_json(
                ("SWN reaches 87.1% accuracy on Adult.", ["S1"]),
                ("Graph Gadgets do not model 3D conformations.", ["S2", "S99"]),
                ("Invented claim.", ["S42"]),
            ),
            SYNTHESIS,
        ]
    )
    chain = build_research_chain(ScopedRetriever(store, k_final=3), llm, AUX, k_final=6)
    result = chain.invoke(
        {
            "question": "What gaps exist across these papers?",
            "paper_ids": [widget.paper_id, gadget.paper_id],
        }
    )

    assert result.status == "ok"
    assert [e.id for e in result.evidence] == ["E1", "E2"]  # invented claim dropped
    assert all(e.citations for e in result.evidence)
    assert [a.statement for a in result.analysis] == ["Both models are only tested on small data."]
    assert result.directions[0].based_on == ["E2"]
    assert result.dropped == {"evidence": 1, "analysis": 1, "directions": 0}
    assert {d.metadata["paper_id"] for d in result.sources} == {widget.paper_id, gadget.paper_id}
    assert len(result.sources) <= 6

    # Chain B sees the evidence list, never the raw excerpts.
    synthesis_prompt = llm.prompts[1][-1].content
    assert "E1 [" in synthesis_prompt and "[S1]" not in synthesis_prompt
    assert result.to_dict()["evidence"][0]["citations"][0]["page"] >= 1


def test_no_valid_evidence_skips_synthesis(indexed):
    store, widget, _ = indexed
    llm = RecordingFakeLLM(responses=[evidence_json(("Made up.", ["S77"]))])
    result = build_research_chain(ScopedRetriever(store), llm, AUX).invoke(
        {"question": "gaps?", "paper_ids": [widget.paper_id]}
    )
    assert result.status == "no_evidence" and not result.analysis
    assert len(llm.prompts) == 1  # synthesis never called


def test_unparseable_synthesis_keeps_evidence(indexed):
    store, widget, _ = indexed
    llm = FakeListChatModel(
        responses=[evidence_json(("SWN uses AdamW.", ["S1"])), "not json", "still not json"]
    )
    result = build_research_chain(ScopedRetriever(store), llm, AUX).invoke(
        {"question": "gaps?", "paper_ids": [widget.paper_id]}
    )
    assert result.status == "unstructured"
    assert result.evidence and result.raw_text == "still not json"


def test_empty_scope_returns_no_evidence(store):
    llm = RecordingFakeLLM(responses=["unused"])
    result = build_research_chain(ScopedRetriever(store), llm, AUX).invoke(
        {"question": "gaps?", "paper_ids": ["missing"]}
    )
    assert result.status == "no_evidence" and llm.prompts == []
