"""LangGraph routing with stub pipelines and a scripted router LLM (no network)."""

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from researchhelp.graph.builder import build_graph

pytestmark = pytest.mark.integration


class RecordingFakeLLM(FakeListChatModel):
    prompts: list = []

    def _call(self, messages, *args, **kwargs):
        self.prompts.append(messages)
        return super()._call(messages, *args, **kwargs)


def stub(name):
    calls = []

    def run(inputs):
        calls.append(inputs)
        return {"pipeline": name, "question": inputs["question"]}

    return RunnableLambda(run), calls


def make(router_responses):
    evidence, ev_calls = stub("evidence")
    research, rs_calls = stub("research")
    router = RecordingFakeLLM(responses=router_responses) if router_responses is not None else None
    return build_graph(evidence, research, router), router, ev_calls, rs_calls


STATE = {"question": "What gaps exist?", "paper_ids": ["a", "b"], "paper_titles": {"a": "A"}}


def test_forced_mode_skips_classifier():
    graph, router, ev_calls, rs_calls = make(['{"intent": "evidence", "reason": "x"}'])
    out = graph.invoke({**STATE, "mode": "research"})
    assert out["result"]["pipeline"] == "research"
    assert (out["intent"], out["route_source"]) == ("research", "forced")
    assert router.prompts == [] and ev_calls == []
    assert rs_calls[0]["paper_ids"] == ["a", "b"]  # scope passed through unchanged


def test_auto_mode_follows_llm_router():
    graph, router, ev_calls, rs_calls = make(['{"intent": "evidence", "reason": "lookup"}'])
    out = graph.invoke({**STATE, "mode": "auto"})
    assert out["result"]["pipeline"] == "evidence"
    assert (out["intent"], out["route_source"], out["route_reason"]) == (
        "evidence",
        "llm",
        "lookup",
    )
    assert len(router.prompts) == 1 and rs_calls == []


def test_invalid_router_output_falls_back_to_keywords():
    graph, router, _, _ = make(['{"intent": "maybe"}', "not json"])
    out = graph.invoke({**STATE, "question": "What research gaps exist?", "mode": "auto"})
    assert (out["intent"], out["route_source"]) == ("research", "keyword")


def test_no_router_llm_uses_keywords():
    graph, _, _, _ = make(None)
    out = graph.invoke({**STATE, "question": "Which datasets were used?"})  # mode defaults auto
    assert (out["intent"], out["route_source"]) == ("evidence", "keyword")


def test_invalid_mode_rejected():
    graph, _, _, _ = make(None)
    with pytest.raises(ValueError):
        graph.invoke({**STATE, "mode": "magic"})
