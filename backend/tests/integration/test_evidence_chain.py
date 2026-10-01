"""Query -> retrieval -> RAG, with a scripted fake LLM (no network)."""

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from researchhelp.rag.evidence.chain import NO_CONTEXT_ANSWER, build_evidence_chain
from researchhelp.rag.evidence.prompts import EVIDENCE_PROMPT
from researchhelp.retrieval.retriever import ScopedRetriever

pytestmark = pytest.mark.integration


class RecordingFakeLLM(FakeListChatModel):
    """Fake chat model that also records the prompts it received."""

    prompts: list = []

    def _call(self, messages, *args, **kwargs):
        self.prompts.append(messages)
        return super()._call(messages, *args, **kwargs)


def test_single_paper_answer_with_resolved_citations(indexed):
    store, widget, _ = indexed
    llm = RecordingFakeLLM(
        responses=["SWN is trained with AdamW at 3e-4 [S1] on Adult [S1] [S42]."]
    )
    chain = build_evidence_chain(ScopedRetriever(store, k_final=3), llm)

    result = chain.invoke(
        {"question": "Which optimizer and learning rate were used?", "paper_ids": [widget.paper_id]}
    )

    assert "[S42]" not in result.answer  # fabricated ID removed
    assert len(result.citations) == 1
    cite = result.citations[0]
    assert cite.paper_id == widget.paper_id
    assert (cite.page, cite.section) == (3, "Experiments")
    assert "AdamW" in cite.snippet

    human = llm.prompts[0][-1].content
    assert "Question: Which optimizer and learning rate were used?" in human
    assert "- Sparse Widget Networks for Tabular Data" in human
    header = "[S1] (Paper: Sparse Widget Networks for Tabular Data | Page 3 | Section: Experiments)"
    assert header in human


def test_multi_paper_prompt_contains_both_papers(indexed):
    store, widget, gadget = indexed
    llm = RecordingFakeLLM(responses=["Comparison [S1] [S2]."])
    chain = build_evidence_chain(ScopedRetriever(store, k_final=4), llm)
    result = chain.invoke(
        {
            "question": "Compare the datasets used in these papers.",
            "paper_ids": [widget.paper_id, gadget.paper_id],
        }
    )
    assert {d.metadata["paper_id"] for d in result.sources} == {widget.paper_id, gadget.paper_id}
    human = llm.prompts[0][-1].content
    assert "Graph Gadgets for Molecular Property Prediction" in human
    assert "Sparse Widget Networks for Tabular Data" in human


def test_no_retrieved_context_skips_the_llm(store, widget_pdf):
    # Empty store: nothing to retrieve for this paper id.
    llm = RecordingFakeLLM(responses=["should not be called"])
    chain = build_evidence_chain(ScopedRetriever(store), llm)
    result = chain.invoke({"question": "anything", "paper_ids": ["missing"]})
    assert result.answer == NO_CONTEXT_ANSWER and result.citations == []
    assert llm.prompts == []


def test_prompt_contains_grounding_rules():
    system = EVIDENCE_PROMPT.messages[0].prompt.template
    assert "ONLY" in system and "[S2]" in system and "say so plainly" in system
