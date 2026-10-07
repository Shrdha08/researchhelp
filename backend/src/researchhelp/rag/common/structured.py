"""Structured (JSON) output from open models: JSON mode + Pydantic validation + one repair retry.

Why not tool calling / ``with_structured_output``: open models served on Groq break tool-call
formats more often than they break plain JSON, and JSON mode works identically with LangChain's
fake chat models in tests. Validation errors are sent back once so the model can fix its output.
"""

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ValidationError


class StructuredOutputError(RuntimeError):
    def __init__(self, message: str, raw: str):
        super().__init__(message)
        self.raw = raw


def extract_json(text: str) -> dict:
    """Parse the outermost JSON object in ``text`` (tolerates code fences and stray prose)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    return json.loads(text[start : end + 1])


def json_mode(llm: BaseChatModel):
    """Ask the provider for a JSON object (Groq/OpenAI ``response_format``)."""
    return llm.bind(response_format={"type": "json_object"})


def invoke_json[T: BaseModel](
    llm,
    messages: list[BaseMessage],
    schema: type[T],
    config: RunnableConfig | None = None,
) -> T:
    """Invoke ``llm`` (already in JSON mode) and validate the reply against ``schema``.

    On invalid output, the error is sent back once for a repair attempt; a second failure raises
    ``StructuredOutputError`` carrying the raw text so callers can degrade gracefully."""
    history = list(messages)
    raw = ""
    for attempt in range(2):
        raw = str(llm.invoke(history, config=config).content)
        try:
            return schema.model_validate(extract_json(raw))
        except (ValueError, ValidationError) as exc:
            if attempt == 1:
                raise StructuredOutputError(f"invalid structured output: {exc}", raw) from exc
            history += [
                AIMessage(raw),
                HumanMessage(
                    "Your reply was not valid JSON for the required schema. Error:\n"
                    f"{str(exc)[:800]}\n\nReturn only the corrected JSON object."
                ),
            ]
    raise AssertionError("unreachable")
