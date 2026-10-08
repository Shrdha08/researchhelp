"""Query use case: validate scope, run the LangGraph app, record the exchange in a conversation.

The graph is created lazily on first use, so the API can start (and /health can report) even
when the LLM is not configured; queries then fail with a clear 503 instead of a crash."""

import threading
from collections.abc import Callable

from researchhelp.rag.common.llm import MissingAPIKeyError
from researchhelp.repository.conversations import ConversationRecord, SqlConversationRepository
from researchhelp.services.errors import ConversationNotFound, LLMUnavailable
from researchhelp.services.paper_service import PaperService

TITLE_CHARS = 80


def state_to_payload(state: dict) -> dict:
    """The structured answer shared by the API response and the stored message history:
    ``{intent, route_source, route_reason, evidence | research}``."""
    body_key = "evidence" if state["intent"] == "evidence" else "research"
    return {
        "intent": state["intent"],
        "route_source": state["route_source"],
        "route_reason": state["route_reason"],
        body_key: state["result"].to_dict(),
    }


def answer_text(state: dict) -> str:
    """Plain-text summary of an answer, stored as the message content."""
    result = state["result"]
    if state["intent"] == "evidence":
        return result.answer
    return (
        f"Research assistance ({result.status}): {len(result.evidence)} evidence items, "
        f"{len(result.analysis)} analysis items, {len(result.directions)} proposed directions."
    )


class QueryService:
    def __init__(
        self,
        graph_factory: Callable,
        papers: PaperService,
        conversations: SqlConversationRepository | None = None,
    ):
        self._graph_factory = graph_factory
        self._graph = None
        self._lock = threading.Lock()
        self.papers = papers
        self.conversations = conversations

    def graph(self):
        with self._lock:
            if self._graph is None:
                try:
                    self._graph = self._graph_factory()
                except MissingAPIKeyError as exc:
                    raise LLMUnavailable(str(exc)) from exc
            return self._graph

    def ask(
        self,
        question: str,
        paper_ids: list[str],
        mode: str = "auto",
        conversation_id: str | None = None,
    ) -> tuple[dict, str | None]:
        """Returns (payload, conversation_id). Without ``conversation_id`` a new conversation is
        started (titled after the question). History is stored only after a successful answer."""
        if (
            conversation_id
            and self.conversations
            and not self.conversations.exists(conversation_id)
        ):
            raise ConversationNotFound(f"conversation {conversation_id} not found")
        titles = self.papers.ready_titles(paper_ids)
        state = self.graph().invoke(
            {"question": question, "paper_ids": list(titles), "paper_titles": titles, "mode": mode}
        )
        payload = state_to_payload(state)
        if self.conversations is None:
            return payload, None
        if conversation_id is None:
            conversation_id = self.conversations.create(question[:TITLE_CHARS]).id
        self.conversations.add_exchange(
            conversation_id,
            question=question,
            mode=mode,
            paper_ids=list(titles),
            answer_text=answer_text(state),
            intent=state["intent"],
            payload=payload,
        )
        return payload, conversation_id

    # ----- history ---------------------------------------------------------------------------

    def list_conversations(self) -> list[ConversationRecord]:
        return self.conversations.list_all() if self.conversations else []

    def get_conversation(self, conversation_id: str) -> ConversationRecord:
        record = self.conversations.get(conversation_id) if self.conversations else None
        if record is None:
            raise ConversationNotFound(f"conversation {conversation_id} not found")
        return record

    def delete_conversation(self, conversation_id: str) -> None:
        if not (self.conversations and self.conversations.delete(conversation_id)):
            raise ConversationNotFound(f"conversation {conversation_id} not found")
