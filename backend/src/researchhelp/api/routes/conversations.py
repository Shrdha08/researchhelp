from typing import Annotated

from fastapi import APIRouter, Depends, status

from researchhelp.api.deps import query_service
from researchhelp.api.schemas import ConversationOut, ConversationSummary, MessageOut, QueryResponse
from researchhelp.repository.conversations import MessageRecord
from researchhelp.services.query_service import QueryService

router = APIRouter(prefix="/conversations", tags=["conversations"])
Service = Annotated[QueryService, Depends(query_service)]


def _message(m: MessageRecord) -> MessageOut:
    data = m.to_dict()
    data["payload"] = QueryResponse.model_validate(m.payload) if m.payload else None
    return MessageOut.model_validate(data)


@router.get("", response_model=list[ConversationSummary])
def list_conversations(service: Service):
    """Conversations, most recently active first."""
    return [ConversationSummary.model_validate(vars(c)) for c in service.list_conversations()]


@router.get("/{conversation_id}", response_model=ConversationOut)
def get_conversation(conversation_id: str, service: Service):
    """A conversation with all its messages. Assistant messages carry the full structured answer
    (evidence/research body and citations), so a UI can re-render history exactly."""
    c = service.get_conversation(conversation_id)
    return ConversationOut(
        id=c.id,
        title=c.title,
        created_at=c.created_at,
        updated_at=c.updated_at,
        message_count=c.message_count,
        messages=[_message(m) for m in c.messages],
    )


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: str, service: Service):
    service.delete_conversation(conversation_id)
