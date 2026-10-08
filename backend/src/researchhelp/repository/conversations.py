"""Conversation and message storage.

Conversation history is stored and displayed only. It is never fed back into the LLM
(conversational memory is a future extension), so every question is answered independently."""

import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from researchhelp.db.models import Conversation, Message


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.isoformat(timespec="seconds")


@dataclass
class MessageRecord:
    id: int
    conversation_id: str
    role: str
    content: str
    created_at: str
    intent: str | None = None
    mode: str | None = None
    paper_ids: list[str] = field(default_factory=list)
    payload: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConversationRecord:
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0
    messages: list[MessageRecord] = field(default_factory=list)


def _message(row: Message) -> MessageRecord:
    return MessageRecord(
        id=row.id,
        conversation_id=row.conversation_id,
        role=row.role,
        content=row.content,
        created_at=_iso(row.created_at),
        intent=row.intent,
        mode=row.mode,
        paper_ids=list(row.paper_ids or []),
        payload=row.payload,
    )


class SqlConversationRepository:
    def __init__(self, sessions: sessionmaker[Session]):
        self._sessions = sessions

    def create(self, title: str) -> ConversationRecord:
        now = datetime.now(UTC)
        row = Conversation(id=str(uuid.uuid4()), title=title[:200], created_at=now, updated_at=now)
        with self._sessions.begin() as s:
            s.add(row)
        return ConversationRecord(row.id, row.title, _iso(now), _iso(now))

    def exists(self, conversation_id: str) -> bool:
        with self._sessions() as s:
            return s.get(Conversation, conversation_id) is not None

    def get(self, conversation_id: str) -> ConversationRecord | None:
        with self._sessions() as s:
            row = s.get(Conversation, conversation_id)
            if row is None:
                return None
            messages = [_message(m) for m in row.messages]
            return ConversationRecord(
                row.id,
                row.title,
                _iso(row.created_at),
                _iso(row.updated_at),
                len(messages),
                messages,
            )

    def list_all(self) -> list[ConversationRecord]:
        """Most recently active first; message counts without loading the messages."""
        with self._sessions() as s:
            counts = dict(
                s.execute(
                    select(Message.conversation_id, func.count(Message.id)).group_by(
                        Message.conversation_id
                    )
                ).all()
            )
            rows = s.scalars(select(Conversation).order_by(Conversation.updated_at.desc()))
            return [
                ConversationRecord(
                    r.id, r.title, _iso(r.created_at), _iso(r.updated_at), counts.get(r.id, 0)
                )
                for r in rows
            ]

    def add_exchange(
        self,
        conversation_id: str,
        question: str,
        mode: str,
        paper_ids: list[str],
        answer_text: str,
        intent: str,
        payload: dict,
    ) -> list[MessageRecord]:
        """Store a question and its answer atomically, so history never has a dangling half."""
        now = datetime.now(UTC)
        with self._sessions.begin() as s:
            conversation = s.get(Conversation, conversation_id, with_for_update=True)
            if conversation is None:
                raise KeyError(conversation_id)
            user = Message(
                conversation_id=conversation_id,
                role="user",
                content=question,
                mode=mode,
                paper_ids=paper_ids,
                created_at=now,
            )
            assistant = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=answer_text,
                intent=intent,
                paper_ids=paper_ids,
                payload=payload,
                created_at=now,
            )
            s.add_all([user, assistant])
            conversation.updated_at = now
            s.flush()
            return [_message(user), _message(assistant)]

    def delete(self, conversation_id: str) -> bool:
        with self._sessions.begin() as s:
            row = s.get(Conversation, conversation_id)
            if row is None:
                return False
            s.delete(row)
            return True
