"""Evidence-based Q&A pipeline (LangChain LCEL).

{question, paper_ids, paper_titles?}
  -> retrieve (ScopedRetriever: paper_id filter, per-paper fan-out)
  -> format [S#] context
  -> prompt | LLM | str
  -> resolve [S#] citations against the retrieved chunks
  -> EvidenceAnswer
"""

from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda

from researchhelp.rag.common.context import (
    Citation,
    format_context,
    format_paper_list,
    resolve_citations,
)
from researchhelp.rag.evidence.prompts import EVIDENCE_PROMPT
from researchhelp.retrieval.retriever import ScopedRetriever

NO_CONTEXT_ANSWER = (
    "I could not find any relevant passages in the selected papers to answer this question."
)


@dataclass
class EvidenceAnswer:
    question: str
    paper_ids: list[str]
    answer: str
    citations: list[Citation] = field(default_factory=list)
    sources: list[Document] = field(default_factory=list)  # everything shown to the LLM

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "paper_ids": self.paper_ids,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "sources": [{"text": d.page_content, **d.metadata} for d in self.sources],
        }


def build_evidence_chain(
    retriever: ScopedRetriever, llm: BaseChatModel
) -> Runnable[dict, EvidenceAnswer]:
    generate = EVIDENCE_PROMPT | llm | StrOutputParser()

    def retrieve(inputs: dict) -> dict:
        docs = retriever.retrieve(inputs["question"], inputs["paper_ids"])
        context, sources = format_context(docs)
        # Titles of all selected papers, so the model can say which ones lack evidence.
        titles = dict(inputs.get("paper_titles") or {})
        for doc in docs:
            titles.setdefault(doc.metadata["paper_id"], doc.metadata["paper_title"])
        return {**inputs, "docs": docs, "context": context, "sources": sources, "titles": titles}

    def answer(state: dict, config: RunnableConfig) -> EvidenceAnswer:
        result = EvidenceAnswer(
            question=state["question"],
            paper_ids=list(state["paper_ids"]),
            answer=NO_CONTEXT_ANSWER,
            sources=state["docs"],
        )
        if not state["docs"]:
            return result  # nothing retrieved: do not let the LLM answer from memory
        raw = generate.invoke(
            {
                "papers": format_paper_list(state["titles"]),
                "context": state["context"],
                "question": state["question"],
            },
            config=config,
        )
        result.answer, result.citations = resolve_citations(raw, state["sources"])
        return result

    return (RunnableLambda(retrieve) | RunnableLambda(answer)).with_config(run_name="evidence_rag")
