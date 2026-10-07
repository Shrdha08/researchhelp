"""Research-assistant pipeline (LangChain LCEL).

{question, paper_ids, paper_titles?}
  -> retrieve twice with the shared ScopedRetriever: the user's question and a fixed
     "limitations / future work" query; merge, de-duplicate, cap
  -> Chain A: extract cited evidence (JSON)  -> drop claims without a valid [S#] source
  -> Chain B: synthesise analysis + directions from the evidence ONLY (JSON)
              -> drop items that do not build on a valid evidence ID
  -> ResearchAnswer (status ok | no_evidence | unstructured)
"""

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda

from researchhelp.rag.common.context import citations_for, format_context, format_paper_list
from researchhelp.rag.common.structured import StructuredOutputError, invoke_json, json_mode
from researchhelp.rag.research import prompts
from researchhelp.rag.research.schemas import (
    EvidenceExtraction,
    EvidenceItem,
    ResearchAnswer,
    Synthesis,
    normalize_ids,
)
from researchhelp.retrieval.retriever import ScopedRetriever

NO_EVIDENCE_MESSAGE = (
    "No relevant evidence was found in the selected papers, so no analysis or research "
    "directions were generated."
)


def merge_ranked(lists: Sequence[Sequence[Document]], limit: int) -> list[Document]:
    """Interleave ranked lists (1st of each, then 2nd of each, ...), dropping duplicate chunks,
    so neither query's results crowd out the other's."""
    merged: list[Document] = []
    seen: set[str] = set()
    for rank in range(max((len(docs) for docs in lists), default=0)):
        for docs in lists:
            if rank < len(docs) and docs[rank].metadata["chunk_id"] not in seen:
                seen.add(docs[rank].metadata["chunk_id"])
                merged.append(docs[rank])
    return merged[:limit]


def validate_evidence(
    extraction: EvidenceExtraction, sources: dict[str, Document]
) -> list[EvidenceItem]:
    """Keep only claims citing at least one excerpt that was really shown; number them E1..En."""
    items: list[EvidenceItem] = []
    for raw in extraction.evidence:
        citations = citations_for(normalize_ids(raw.sources, "S"), sources)
        if not citations or not raw.claim.strip():
            continue
        items.append(
            EvidenceItem(
                id=f"E{len(items) + 1}",
                claim=raw.claim.strip(),
                paper_title=citations[0].paper_title,
                citations=citations,
            )
        )
    return items


def validate_synthesis(synthesis: Synthesis, evidence_ids: set[str]) -> Synthesis:
    """Drop analysis/direction items that do not build on at least one real evidence item."""

    def keep(items):
        kept = []
        for item in items:
            links = [e for e in normalize_ids(item.based_on, "E") if e in evidence_ids]
            if links:
                kept.append(item.model_copy(update={"based_on": links}))
        return kept

    return Synthesis(analysis=keep(synthesis.analysis), directions=keep(synthesis.directions))


def format_evidence(items: Sequence[EvidenceItem]) -> str:
    return "\n".join(f"{e.id} [{e.paper_title}, p.{e.citations[0].page}] {e.claim}" for e in items)


def build_research_chain(
    retriever: ScopedRetriever,
    llm: BaseChatModel,
    aux_query: str,
    k_final: int = 10,
) -> Runnable[dict, ResearchAnswer]:
    model = json_mode(llm)

    def retrieve(inputs: dict) -> dict:
        question, paper_ids = inputs["question"], inputs["paper_ids"]
        docs = merge_ranked(
            [retriever.retrieve(question, paper_ids), retriever.retrieve(aux_query, paper_ids)],
            limit=k_final,
        )
        titles = dict(inputs.get("paper_titles") or {})
        for doc in docs:
            titles.setdefault(doc.metadata["paper_id"], doc.metadata["paper_title"])
        return {**inputs, "docs": docs, "titles": titles}

    def answer(state: dict, config: RunnableConfig) -> ResearchAnswer:
        result = ResearchAnswer(
            question=state["question"],
            paper_ids=list(state["paper_ids"]),
            status="no_evidence",
            sources=state["docs"],
            message=NO_EVIDENCE_MESSAGE,
        )
        if not state["docs"]:
            return result

        context, sources = format_context(state["docs"])
        try:
            extraction = invoke_json(
                model,
                [
                    SystemMessage(prompts.EVIDENCE_SYSTEM),
                    HumanMessage(
                        prompts.EVIDENCE_HUMAN.format(
                            papers=format_paper_list(state["titles"]),
                            context=context,
                            question=state["question"],
                        )
                    ),
                ],
                EvidenceExtraction,
                config,
            )
        except StructuredOutputError as exc:
            result.status, result.raw_text = "unstructured", exc.raw
            result.message = "The model's evidence output could not be parsed; showing raw text."
            return result

        result.evidence = validate_evidence(extraction, sources)
        if not result.evidence:
            return result  # never speculate without evidence

        try:
            synthesis = invoke_json(
                model,
                [
                    SystemMessage(prompts.SYNTHESIS_SYSTEM),
                    HumanMessage(
                        prompts.SYNTHESIS_HUMAN.format(
                            question=state["question"], evidence=format_evidence(result.evidence)
                        )
                    ),
                ],
                Synthesis,
                config,
            )
        except StructuredOutputError as exc:
            result.status, result.raw_text = "unstructured", exc.raw
            result.message = "Evidence is shown; the analysis output could not be parsed."
            return result

        synthesis = validate_synthesis(synthesis, {e.id for e in result.evidence})
        result.analysis, result.directions = synthesis.analysis, synthesis.directions
        result.status, result.message = "ok", ""
        return result

    return (RunnableLambda(retrieve) | RunnableLambda(answer)).with_config(
        run_name="research_assistant"
    )
