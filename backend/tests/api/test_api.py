"""API tests: the real FastAPI app (lifespan, background ingestion, error mapping) with an
in-memory Qdrant, hashing encoders, a temporary JSON registry and scripted fake LLMs."""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from researchhelp.api.main import create_app
from researchhelp.graph.builder import build_graph
from researchhelp.rag.common.llm import MissingAPIKeyError
from researchhelp.rag.evidence.chain import build_evidence_chain
from researchhelp.rag.research.chain import build_research_chain
from researchhelp.repository.conversations import SqlConversationRepository
from researchhelp.repository.sql_repository import SqlPaperRepository
from researchhelp.retrieval.retriever import ScopedRetriever
from researchhelp.services.paper_service import PaperService
from researchhelp.services.query_service import QueryService

pytestmark = pytest.mark.integration


@pytest.fixture
def make_client(store, tmp_path, sqlite_sessions):
    """Returns a factory: make_client(llm_responses) -> (TestClient, PaperService)."""

    def make(responses=None, graph_factory=None):
        papers = PaperService(
            SqlPaperRepository(sqlite_sessions),
            store,
            tmp_path / "uploads",
            chunk_size=300,
            chunk_overlap=50,
            max_upload_bytes=2 * 1024 * 1024,
        )

        def default_graph():
            llm = FakeListChatModel(responses=responses or ["unused"])
            retriever = ScopedRetriever(store, k_final=3)
            return build_graph(
                build_evidence_chain(retriever, llm),
                build_research_chain(retriever, llm, "limitations future work", k_final=6),
                router_llm=None,  # keyword routing
            )

        conversations = SqlConversationRepository(sqlite_sessions)
        service = QueryService(graph_factory or default_graph, papers, conversations)
        app = create_app(papers, service)
        return TestClient(app), papers

    return make


def upload(client, *paths):
    files = [("files", (p.name, p.read_bytes(), "application/pdf")) for p in paths]
    return client.post("/papers/upload", files=files)


def test_health(make_client):
    client, _ = make_client()
    with client:
        body = client.get("/health").json()
    assert body == {
        "status": "ok",
        "qdrant": True,
        "database": True,
        "papers": 0,
        "llm_configured": True,
    }


def test_upload_ingests_in_background_and_lists(make_client, widget_pdf, gadget_pdf):
    client, papers = make_client()
    with client:
        res = upload(client, widget_pdf, gadget_pdf)
        assert res.status_code == 202
        items = res.json()["papers"]
        assert [i["duplicate"] for i in items] == [False, False]
        assert items[0]["paper"]["status"] == "processing"  # response is sent before ingestion

        pid = items[0]["paper"]["id"]
        paper = client.get(f"/papers/{pid}").json()  # background task has run by now
        assert paper["status"] == "ready"
        assert paper["title"] == "Sparse Widget Networks for Tabular Data"
        assert paper["num_pages"] == 3 and paper["num_chunks"] > 0
        assert papers.store.count(pid) == paper["num_chunks"]
        assert len(client.get("/papers").json()) == 2


def test_duplicate_upload_is_not_reindexed(make_client, widget_pdf):
    client, papers = make_client()
    with client:
        first = upload(client, widget_pdf).json()["papers"][0]["paper"]
        second = upload(client, widget_pdf).json()["papers"][0]
    assert second["duplicate"] is True and second["paper"]["id"] == first["id"]
    assert len(papers.list_all()) == 1


def test_invalid_upload_rejected_and_nothing_stored(make_client, widget_pdf, tmp_path):
    client, papers = make_client()
    fake = tmp_path / "notes.pdf"
    fake.write_bytes(b"just text, not a pdf")
    with client:
        res = upload(client, widget_pdf, fake)  # one bad file rejects the whole request
        assert res.status_code == 400 and "not a PDF" in res.json()["detail"]
        assert (
            client.post(
                "/papers/upload", files=[("files", ("a.txt", b"x", "text/plain"))]
            ).status_code
            == 400
        )
    assert papers.list_all() == []


def test_corrupt_pdf_fails_cleanly_and_cannot_be_queried(make_client, tmp_path):
    client, _ = make_client()
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.7\n this is not really a pdf")
    with client:
        pid = upload(client, broken).json()["papers"][0]["paper"]["id"]
        paper = client.get(f"/papers/{pid}").json()
        assert paper["status"] == "failed" and paper["error"]
        res = client.post("/query", json={"question": "anything?", "paper_ids": [pid]})
    assert res.status_code == 409


def test_query_evidence_returns_cited_answer(make_client, widget_pdf):
    client, _ = make_client(responses=["SWN is trained with AdamW [S1]."])
    with client:
        pid = upload(client, widget_pdf).json()["papers"][0]["paper"]["id"]
        res = client.post(
            "/query", json={"question": "Which optimizer was used?", "paper_ids": [pid]}
        )
    assert res.status_code == 200, res.text
    body = res.json()
    assert (body["intent"], body["route_source"]) == ("evidence", "keyword")
    assert body["research"] is None
    assert body["evidence"]["answer"] == "SWN is trained with AdamW [S1]."
    cite = body["evidence"]["citations"][0]
    assert cite["paper_id"] == pid and cite["page"] >= 1
    assert body["evidence"]["sources"]


def test_query_research_mode_returns_three_sections(make_client, widget_pdf):
    evidence = json.dumps({"evidence": [{"claim": "SWN uses AdamW.", "sources": ["S1"]}]})
    synthesis = json.dumps(
        {
            "analysis": [{"statement": "Only one optimizer tested.", "based_on": ["E1"]}],
            "directions": [
                {
                    "title": "Optimizer study",
                    "rationale": "r",
                    "based_on": ["E1"],
                    "validation_experiment": "Compare SGD and AdamW.",
                }
            ],
        }
    )
    client, _ = make_client(responses=[evidence, synthesis])
    with client:
        pid = upload(client, widget_pdf).json()["papers"][0]["paper"]["id"]
        res = client.post(
            "/query",
            json={"question": "What could be improved?", "paper_ids": [pid], "mode": "research"},
        )
    body = res.json()
    assert (body["intent"], body["route_source"]) == ("research", "forced")
    research = body["research"]
    assert research["status"] == "ok"
    assert research["evidence"][0]["citations"][0]["page"] >= 1
    assert research["analysis"][0]["based_on"] == ["E1"]
    assert research["directions"][0]["validation_experiment"] == "Compare SGD and AdamW."


def test_query_validation_and_unknown_papers(make_client):
    client, _ = make_client()
    with client:
        assert client.post("/query", json={"question": "q?", "paper_ids": []}).status_code == 422
        bad_mode = {"question": "hello?", "paper_ids": ["x"], "mode": "magic"}
        assert client.post("/query", json=bad_mode).status_code == 422
        res = client.post("/query", json={"question": "hello?", "paper_ids": ["nope"]})
    assert res.status_code == 404


def test_query_without_llm_key_returns_503(make_client, widget_pdf):
    def no_key():
        raise MissingAPIKeyError("GROQ_API_KEY is not set")

    client, _ = make_client(graph_factory=no_key)
    with client:
        pid = upload(client, widget_pdf).json()["papers"][0]["paper"]["id"]
        res = client.post("/query", json={"question": "Which data?", "paper_ids": [pid]})
    assert res.status_code == 503 and "GROQ_API_KEY" in res.json()["detail"]


def test_delete_removes_index_file_and_record(make_client, widget_pdf):
    client, papers = make_client()
    with client:
        pid = upload(client, widget_pdf).json()["papers"][0]["paper"]["id"]
        assert papers.path_for(pid).exists()
        assert client.delete(f"/papers/{pid}").status_code == 204
        assert client.get(f"/papers/{pid}").status_code == 404
        assert client.delete(f"/papers/{pid}").status_code == 404
    assert papers.store.count(pid) == 0
    assert not papers.path_for(pid).exists()


def ask(client, pid, question="Which optimizer was used?", **extra):
    return client.post("/query", json={"question": question, "paper_ids": [pid], **extra})


def test_query_creates_conversation_and_follow_up_appends(make_client, widget_pdf):
    client, _ = make_client(responses=["AdamW [S1].", "3e-4 [S1]."])
    with client:
        pid = upload(client, widget_pdf).json()["papers"][0]["paper"]["id"]
        first = ask(client, pid).json()
        cid = first["conversation_id"]
        assert cid
        second = ask(client, pid, "And the learning rate?", conversation_id=cid).json()
        assert second["conversation_id"] == cid

        listed = client.get("/conversations").json()
        assert [(c["id"], c["message_count"]) for c in listed] == [(cid, 4)]
        assert listed[0]["title"] == "Which optimizer was used?"

        history = client.get(f"/conversations/{cid}").json()
    roles = [m["role"] for m in history["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    user_msg, answer_msg = history["messages"][:2]
    assert (user_msg["content"], user_msg["mode"], user_msg["paper_ids"]) == (
        "Which optimizer was used?",
        "auto",
        [pid],
    )
    # The stored payload re-renders the answer exactly as /query returned it, citations included.
    assert answer_msg["payload"]["evidence"]["answer"] == first["evidence"]["answer"]
    assert answer_msg["payload"]["evidence"]["citations"] == first["evidence"]["citations"]
    assert answer_msg["intent"] == "evidence"


def test_research_answer_is_stored_with_all_sections(make_client, widget_pdf):
    evidence = json.dumps({"evidence": [{"claim": "SWN uses AdamW.", "sources": ["S1"]}]})
    synthesis = json.dumps(
        {
            "analysis": [{"statement": "Single optimizer.", "based_on": ["E1"]}],
            "directions": [
                {"title": "T", "rationale": "r", "based_on": ["E1"], "validation_experiment": "x"}
            ],
        }
    )
    client, _ = make_client(responses=[evidence, synthesis])
    with client:
        pid = upload(client, widget_pdf).json()["papers"][0]["paper"]["id"]
        cid = ask(client, pid, "What could be improved?", mode="research").json()["conversation_id"]
        message = client.get(f"/conversations/{cid}").json()["messages"][1]
    research = message["payload"]["research"]
    assert message["intent"] == "research" and research["status"] == "ok"
    assert research["evidence"][0]["citations"] and research["directions"][0]["based_on"] == ["E1"]
    assert message["content"].startswith("Research assistance (ok): 1 evidence items")


def test_failed_query_is_not_stored(make_client, widget_pdf):
    def no_key():
        raise MissingAPIKeyError("GROQ_API_KEY is not set")

    client, _ = make_client(graph_factory=no_key)
    with client:
        pid = upload(client, widget_pdf).json()["papers"][0]["paper"]["id"]
        assert ask(client, pid).status_code == 503
        assert client.get("/conversations").json() == []


def test_unknown_conversation_is_404_before_any_work(make_client, widget_pdf):
    client, _ = make_client(responses=["unused"])
    with client:
        pid = upload(client, widget_pdf).json()["papers"][0]["paper"]["id"]
        assert ask(client, pid, conversation_id="nope").status_code == 404
        assert client.get("/conversations/nope").status_code == 404
        assert client.delete("/conversations/nope").status_code == 404


def test_delete_conversation(make_client, widget_pdf):
    client, _ = make_client(responses=["AdamW [S1]."])
    with client:
        pid = upload(client, widget_pdf).json()["papers"][0]["paper"]["id"]
        cid = ask(client, pid).json()["conversation_id"]
        assert client.delete(f"/conversations/{cid}").status_code == 204
        assert client.get("/conversations").json() == []
        assert client.get(f"/conversations/{cid}").status_code == 404


def test_deleting_a_paper_keeps_conversation_history(make_client, widget_pdf):
    client, _ = make_client(responses=["AdamW [S1]."])
    with client:
        pid = upload(client, widget_pdf).json()["papers"][0]["paper"]["id"]
        cid = ask(client, pid).json()["conversation_id"]
        assert client.delete(f"/papers/{pid}").status_code == 204
        history = client.get(f"/conversations/{cid}").json()
    assert history["message_count"] == 2  # answers cite pages that were true when asked
