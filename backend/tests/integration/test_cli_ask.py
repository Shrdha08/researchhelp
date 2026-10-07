"""`researchhelp ask` end to end through the graph: real chains, synthetic PDFs, fake LLM."""

import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from typer.testing import CliRunner

import researchhelp.graph.factory as graph_factory
from researchhelp import cli
from researchhelp.graph.builder import build_graph
from researchhelp.rag.evidence.chain import build_evidence_chain
from researchhelp.rag.research.chain import build_research_chain
from researchhelp.retrieval.retriever import ScopedRetriever

pytestmark = pytest.mark.integration


@pytest.fixture
def run_ask(indexed, monkeypatch):
    store, widget, gadget = indexed
    titles = {widget.paper_id: widget.title, gadget.paper_id: gadget.title}

    def run(args, responses):
        llm = FakeListChatModel(responses=responses)
        retriever = ScopedRetriever(store, k_final=3)
        graph = build_graph(
            build_evidence_chain(retriever, llm),
            build_research_chain(retriever, llm, "limitations future work", k_final=6),
            router_llm=None,  # keyword routing
        )
        monkeypatch.setattr(graph_factory, "get_app", lambda strategy=None: graph)
        monkeypatch.setattr(cli, "_resolve_papers", lambda refs: titles)
        return CliRunner().invoke(cli.app, ["ask", *args, "-p", "x"])

    return run


def test_ask_research_mode_prints_three_sections(run_ask):
    evidence = json.dumps({"evidence": [{"claim": "Gadgets ignore 3D.", "sources": ["S1"]}]})
    synthesis = json.dumps(
        {
            "analysis": [{"statement": "Geometry is untested.", "based_on": ["E1"]}],
            "directions": [
                {
                    "title": "Add conformers",
                    "rationale": "Follows from E1.",
                    "based_on": ["E1"],
                    "validation_experiment": "QM9 with 3D inputs.",
                }
            ],
        }
    )
    out = run_ask(["What should be improved?", "--mode", "research"], [evidence, synthesis])
    assert out.exit_code == 0, out.output
    assert "[route: research (forced)" in out.output
    for header in ("== Evidence", "== Analysis (inferred", "== Proposed research directions"):
        assert header in out.output
    assert "E1. Gadgets ignore 3D." in out.output
    assert "based on E1" in out.output and "How to test: QM9 with 3D inputs." in out.output


def test_ask_auto_routes_factual_question_to_evidence(run_ask):
    out = run_ask(["Which datasets were used?"], ["They used QM9 [S1]."])
    assert out.exit_code == 0, out.output
    assert "[route: evidence (keyword)" in out.output
    assert "They used QM9 [S1]." in out.output and "Sources:" in out.output


def test_ask_json_output(run_ask):
    out = run_ask(["Which datasets were used?", "--json"], ["QM9 [S1]."])
    data = json.loads(out.output)
    assert data["intent"] == "evidence" and data["result"]["citations"]


def test_ask_rejects_bad_mode(run_ask):
    out = run_ask(["q", "--mode", "magic"], [])
    assert out.exit_code != 0
