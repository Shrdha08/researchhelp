"""Build the production graph from Settings (one place that wires models and chains)."""

from functools import lru_cache

from researchhelp.config import get_settings
from researchhelp.graph.builder import build_graph
from researchhelp.rag.common.llm import get_chat_model
from researchhelp.rag.evidence.chain import build_evidence_chain
from researchhelp.rag.research.chain import build_research_chain
from researchhelp.retrieval.factory import get_retriever


@lru_cache
def get_app(strategy: str | None = None):
    """Compiled graph; ``strategy`` overrides the retrieval strategy (for comparisons)."""
    settings = get_settings()
    retriever = get_retriever(strategy)
    llm = get_chat_model()
    return build_graph(
        evidence_chain=build_evidence_chain(retriever, llm),
        research_chain=build_research_chain(
            retriever, llm, settings.research_aux_query, settings.research_k_final
        ),
        router_llm=get_chat_model(model=settings.router_model),
    )
