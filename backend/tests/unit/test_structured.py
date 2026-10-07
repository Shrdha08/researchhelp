import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from researchhelp.rag.common.structured import (
    StructuredOutputError,
    extract_json,
    invoke_json,
    json_mode,
)


class Out(BaseModel):
    answer: str
    score: int


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_invoke_json_valid_first_try():
    llm = FakeListChatModel(responses=['{"answer": "x", "score": 3}'])
    assert invoke_json(json_mode(llm), [HumanMessage("q")], Out) == Out(answer="x", score=3)


def test_invoke_json_repairs_once():
    llm = FakeListChatModel(responses=['{"answer": "x"}', '{"answer": "x", "score": 1}'])
    assert invoke_json(llm, [HumanMessage("q")], Out).score == 1


def test_invoke_json_raises_after_second_failure_with_raw_text():
    llm = FakeListChatModel(responses=["not json", "still not json"])
    with pytest.raises(StructuredOutputError) as err:
        invoke_json(llm, [HumanMessage("q")], Out)
    assert err.value.raw == "still not json"
