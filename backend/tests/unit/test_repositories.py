"""One contract, every implementation: the JSON registry, SQLite (via the real migration) and
PostgreSQL must behave identically, because the services cannot tell them apart."""

import time

import pytest

from researchhelp.repository.base import PaperRecord
from researchhelp.repository.conversations import SqlConversationRepository
from researchhelp.repository.json_repository import JsonPaperRepository, import_json_registry
from researchhelp.repository.sql_repository import SqlPaperRepository


@pytest.fixture(params=["json", "sqlite", "postgres"])
def paper_repo(request, tmp_path):
    if request.param == "json":
        return JsonPaperRepository(tmp_path / "papers.json")
    sessions = request.getfixturevalue(
        "sqlite_sessions" if request.param == "sqlite" else "pg_sessions"
    )
    return SqlPaperRepository(sessions)


@pytest.fixture(params=["sqlite", "postgres"])
def conv_repo(request):
    sessions = request.getfixturevalue(
        "sqlite_sessions" if request.param == "sqlite" else "pg_sessions"
    )
    return SqlConversationRepository(sessions)


# ---- papers ------------------------------------------------------------------------------------


def test_paper_add_get_list(paper_repo):
    paper_repo.add(PaperRecord(id="a", filename="a.pdf"))
    time.sleep(0.01)
    paper_repo.add(PaperRecord(id="b", filename="b.pdf", status="ready", title="B", num_pages=3))
    assert paper_repo.get("a").status == "processing"
    assert paper_repo.get("missing") is None
    assert [p.id for p in paper_repo.list_all()] == ["a", "b"]  # oldest first
    assert paper_repo.get("b").title == "B" and paper_repo.get("b").num_pages == 3


def test_paper_duplicate_add_raises_keyerror(paper_repo):
    paper_repo.add(PaperRecord(id="a", filename="a.pdf"))
    with pytest.raises(KeyError):
        paper_repo.add(PaperRecord(id="a", filename="other.pdf"))


def test_paper_update_changes_fields_and_timestamp(paper_repo):
    created = paper_repo.add(PaperRecord(id="a", filename="a.pdf"))
    updated = paper_repo.update("a", status="ready", title="T", num_chunks=9, error=None)
    assert (updated.status, updated.title, updated.num_chunks) == ("ready", "T", 9)
    assert paper_repo.get("a").num_chunks == 9
    assert updated.updated_at >= created.updated_at
    with pytest.raises(KeyError):
        paper_repo.update("missing", status="ready")


def test_paper_delete_is_idempotent(paper_repo):
    paper_repo.add(PaperRecord(id="a", filename="a.pdf"))
    paper_repo.delete("a")
    paper_repo.delete("a")
    assert paper_repo.list_all() == []


def test_import_json_registry_moves_records_once(tmp_path, sqlite_sessions):
    path = tmp_path / "papers.json"
    legacy = JsonPaperRepository(path)
    legacy.add(PaperRecord(id="a", filename="a.pdf", status="ready", title="A"))
    legacy.add(PaperRecord(id="b", filename="b.pdf"))
    target = SqlPaperRepository(sqlite_sessions)
    target.add(PaperRecord(id="b", filename="already-there.pdf"))

    assert import_json_registry(path, target) == 1  # "b" already existed
    assert target.get("a").title == "A" and target.get("b").filename == "already-there.pdf"
    assert not path.exists() and path.with_suffix(".json.imported").exists()
    assert import_json_registry(path, target) == 0


# ---- conversations -----------------------------------------------------------------------------

PAYLOAD = {
    "intent": "evidence",
    "evidence": {
        "answer": "Naïve Bayes → 3e-4 [S1]",
        "citations": [{"page": 5, "snippet": "日本語"}],
    },
}


def exchange(repo, cid, question="Q?", answer="A."):
    return repo.add_exchange(cid, question, "auto", ["p1", "p2"], answer, "evidence", PAYLOAD)


def test_conversation_exchange_round_trip(conv_repo):
    conv = conv_repo.create("What batch size?")
    user, assistant = exchange(conv_repo, conv.id, "What batch size?", "128 [S1].")
    got = conv_repo.get(conv.id)
    assert got.message_count == 2 and [m.role for m in got.messages] == ["user", "assistant"]
    assert (got.messages[0].content, got.messages[0].mode) == ("What batch size?", "auto")
    assert got.messages[0].payload is None and got.messages[0].intent is None
    assert got.messages[1].intent == "evidence" and got.messages[1].paper_ids == ["p1", "p2"]
    assert got.messages[1].payload == PAYLOAD  # nested JSON and unicode survive
    assert user.id < assistant.id


def test_conversation_list_orders_by_recent_activity_with_counts(conv_repo):
    first = conv_repo.create("first")
    conv_repo.create("second")
    time.sleep(0.01)
    exchange(conv_repo, first.id)
    listed = conv_repo.list_all()
    assert [c.title for c in listed] == ["first", "second"]  # "first" was touched last
    assert [c.message_count for c in listed] == [2, 0]


def test_conversation_delete_cascades_to_messages(conv_repo, request):
    conv = conv_repo.create("temp")
    exchange(conv_repo, conv.id)
    assert conv_repo.delete(conv.id) is True and conv_repo.delete(conv.id) is False
    assert conv_repo.get(conv.id) is None
    with conv_repo._sessions() as s:
        from sqlalchemy import func, select

        from researchhelp.db.models import Message

        assert s.scalar(select(func.count(Message.id))) == 0


def test_exchange_for_unknown_conversation_stores_nothing(conv_repo):
    with pytest.raises(KeyError):
        exchange(conv_repo, "no-such-conversation")
    assert conv_repo.list_all() == []
