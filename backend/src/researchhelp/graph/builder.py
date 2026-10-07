"""LangGraph application: route each question to evidence Q&A or the research assistant.

    START ──(mode = evidence | research)──────────────► evidence_rag | research_assistant ─► END
      └───(mode = auto)──► classify ──(intent)────────► evidence_rag | research_assistant ─► END

The graph decides only the *intent*. Scope (which papers) always comes from ``paper_ids``; both
pipelines share the same retriever. All RAG logic lives in ``researchhelp.rag``; nodes only
call the chains.
"""

from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import END, START, StateGraph

from researchhelp.graph.routing import classify_intent
from researchhelp.graph.state import GraphState

EVIDENCE_NODE = "evidence_rag"
RESEARCH_NODE = "research_assistant"
_NODE_FOR = {"evidence": EVIDENCE_NODE, "research": RESEARCH_NODE}


def _chain_input(state: GraphState) -> dict:
    return {
        "question": state["question"],
        "paper_ids": state["paper_ids"],
        "paper_titles": state.get("paper_titles") or {},
    }


def _forced(state: GraphState, intent: str) -> dict:
    """Routing fields for a node reached directly because the user forced the mode."""
    if state.get("intent"):
        return {}
    return {"intent": intent, "route_reason": "selected by the user", "route_source": "forced"}


def build_graph(evidence_chain: Runnable, research_chain: Runnable, router_llm=None):
    def entry(state: GraphState) -> str:
        mode = state.get("mode", "auto")
        if mode == "auto":
            return "classify"
        if mode not in _NODE_FOR:
            raise ValueError(f"mode must be auto, evidence or research, not {mode!r}")
        return _NODE_FOR[mode]

    def classify(state: GraphState, config: RunnableConfig) -> dict:
        intent, reason, source = classify_intent(router_llm, state["question"], config)
        return {"intent": intent, "route_reason": reason, "route_source": source}

    def evidence_rag(state: GraphState, config: RunnableConfig) -> dict:
        return {
            **_forced(state, "evidence"),
            "result": evidence_chain.invoke(_chain_input(state), config),
        }

    def research_assistant(state: GraphState, config: RunnableConfig) -> dict:
        return {
            **_forced(state, "research"),
            "result": research_chain.invoke(_chain_input(state), config),
        }

    graph = StateGraph(GraphState)
    graph.add_node("classify", classify)
    graph.add_node(EVIDENCE_NODE, evidence_rag)
    graph.add_node(RESEARCH_NODE, research_assistant)
    graph.add_conditional_edges(START, entry, ["classify", EVIDENCE_NODE, RESEARCH_NODE])
    graph.add_conditional_edges(
        "classify", lambda s: _NODE_FOR[s["intent"]], [EVIDENCE_NODE, RESEARCH_NODE]
    )
    graph.add_edge(EVIDENCE_NODE, END)
    graph.add_edge(RESEARCH_NODE, END)
    return graph.compile()
