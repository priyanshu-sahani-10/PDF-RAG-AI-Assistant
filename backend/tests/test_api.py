from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import (
    FakeEmbedder,
    FakeOllama,
    make_blank_pdf,
    make_encrypted_pdf,
    make_pdf,
    upload,
)

BIO = ["Photosynthesis converts sunlight into chemical energy inside plant leaves.", "Mitochondria produce energy for animal cells through respiration."]
FIN = ["The stock market rallied as interest rates fell and bond yields dropped sharply."]


def ready(client, doc_id):
    return client.get(f"/api/documents/{doc_id}").json()


def test_health_reports_ollama(client):
    body = client.get("/api/health").json()
    assert body["backend"] is True
    assert body["ollama"]["available"] is True
    assert body["max_upload_mb"] == 1


def test_health_when_ollama_down(settings):
    app = create_app(settings=settings, embedder=FakeEmbedder(), ollama=FakeOllama(available=False))
    body = TestClient(app).get("/api/health").json()
    assert body["ollama"]["available"] is False


def test_upload_index_list_get(client):
    res = upload(client, make_pdf(BIO), "bio.pdf")
    assert res.status_code == 202
    doc_id = res.json()["id"]
    doc = ready(client, doc_id)  # background task has completed by now
    assert doc["status"] == "ready"
    assert doc["page_count"] == 2
    assert doc["chunk_count"] >= 2
    listing = client.get("/api/documents").json()["documents"]
    assert [d["id"] for d in listing] == [doc_id]


def test_duplicate_upload_rejected(client):
    data = make_pdf(BIO)
    assert upload(client, data, "a.pdf").status_code == 202
    res = upload(client, data, "renamed.pdf")
    assert res.status_code == 409
    assert res.json()["error"]["code"] == "duplicate_document"
    assert len(client.get("/api/documents").json()["documents"]) == 1


def test_upload_validation(client):
    assert upload(client, b"hello", "note.txt", "text/plain").status_code == 400
    assert upload(client, b"just text", "fake.pdf").json()["error"]["code"] == "invalid_file_type"
    assert upload(client, b"", "empty.pdf").status_code == 400
    big = make_pdf(["x"]) + b"0" * (1024 * 1024 + 10)
    assert upload(client, big, "big.pdf").status_code == 413


def test_filename_is_sanitized(client):
    res = upload(client, make_pdf(BIO), "../../etc/pass<wd>.pdf")
    name = res.json()["filename"]
    assert "/" not in name and ".." not in name and "<" not in name


def test_failed_documents_show_clear_errors(client):
    blank = upload(client, make_blank_pdf(), "scan.pdf").json()["id"]
    assert ready(client, blank)["status"] == "failed"
    assert "OCR" in ready(client, blank)["error"]
    locked = upload(client, make_encrypted_pdf(), "locked.pdf").json()["id"]
    assert "password" in ready(client, locked)["error"].lower()
    bad = upload(client, b"%PDF-1.4 garbage garbage", "bad.pdf").json()["id"]
    assert ready(client, bad)["status"] == "failed"
    # failed docs leave no vectors behind
    services = client.app.state.services
    for doc_id in (blank, locked, bad):
        assert services.vectors.count_for_document(doc_id) == 0


def test_chat_returns_answer_with_page_citations(client, ollama):
    doc_id = upload(client, make_pdf(BIO), "bio.pdf").json()["id"]
    res = client.post("/api/chat", json={"question": "What do mitochondria produce?"})
    assert res.status_code == 200
    body = res.json()
    assert body["grounded"] is True
    assert body["answer"] == "Fake answer based on the context."
    top = body["sources"][0]
    assert top["filename"] == "bio.pdf"
    assert top["document_id"] == doc_id
    assert top["page_number"] == 2
    assert "Mitochondria" in top["text"]
    assert top["excerpt"]
    assert body["session_id"]
    # retrieved text reached the prompt, and the PDF is flagged untrusted
    system, prompt = ollama.calls[-1]
    assert "untrusted" in system.lower()
    assert "Mitochondria" in prompt


def test_chat_respects_document_scope(client):
    bio = upload(client, make_pdf(BIO), "bio.pdf").json()["id"]
    fin = upload(client, make_pdf(FIN), "fin.pdf").json()["id"]
    res = client.post("/api/chat", json={"question": "interest rates and bond yields", "document_ids": [fin]})
    assert {s["document_id"] for s in res.json()["sources"]} == {fin}
    res = client.post("/api/chat", json={"question": "interest rates and bond yields", "document_ids": [bio]})
    assert all(s["document_id"] == bio for s in res.json()["sources"])


def test_chat_without_relevant_content_skips_llm(client, ollama):
    upload(client, make_pdf(BIO), "bio.pdf")
    res = client.post("/api/chat", json={"question": "zebra quantum saxophone volcano"})
    body = res.json()
    assert body["grounded"] is False
    assert body["sources"] == []
    assert "couldn't find" in body["answer"]
    assert ollama.calls == []


def test_chat_errors(client, settings):
    assert client.post("/api/chat", json={"question": "hi"}).json()["error"]["code"] == "no_documents"
    upload(client, make_pdf(BIO), "bio.pdf")
    assert client.post("/api/chat", json={"question": "   "}).status_code == 422
    res = client.post("/api/chat", json={"question": "hi", "document_ids": ["nope"]})
    assert res.json()["error"]["code"] == "documents_not_ready"
    res = client.post("/api/chat", json={"question": "mitochondria", "session_id": "missing"})
    assert res.status_code == 404


def test_chat_when_ollama_unavailable(settings):
    app = create_app(settings=settings, embedder=FakeEmbedder(), ollama=FakeOllama(available=False))
    client = TestClient(app)
    upload(client, make_pdf(BIO), "bio.pdf")
    res = client.post("/api/chat", json={"question": "What do mitochondria produce?"})
    assert res.status_code == 503
    assert res.json()["error"]["code"] == "ollama_unavailable"


def test_chat_sessions_crud_and_history(client):
    upload(client, make_pdf(BIO), "bio.pdf")
    first = client.post("/api/chat", json={"question": "What do mitochondria produce?"}).json()
    sid = first["session_id"]
    client.post("/api/chat", json={"question": "Tell me about photosynthesis", "session_id": sid})
    detail = client.get(f"/api/chat/sessions/{sid}").json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant", "user", "assistant"]
    assert detail["messages"][1]["sources"][0]["page_number"] == 2
    created = client.post("/api/chat/sessions", json={"title": "Empty chat"})
    assert created.status_code == 201
    assert client.post("/api/chat/sessions").status_code == 201
    ids = [s["id"] for s in client.get("/api/chat/sessions").json()["sessions"]]
    assert sid in ids and len(ids) == 3
    assert client.delete(f"/api/chat/sessions/{sid}").json()["deleted"] is True
    assert client.get(f"/api/chat/sessions/{sid}").status_code == 404
    assert client.delete(f"/api/chat/sessions/{sid}").status_code == 404


def test_delete_document_removes_everything(client, settings):
    doc_id = upload(client, make_pdf(BIO), "bio.pdf").json()["id"]
    services = client.app.state.services
    assert services.vectors.count_for_document(doc_id) > 0
    assert (settings.upload_dir / f"{doc_id}.pdf").exists()
    assert client.delete(f"/api/documents/{doc_id}").json() == {"deleted": True, "id": doc_id}
    assert services.vectors.count_for_document(doc_id) == 0
    assert not (settings.upload_dir / f"{doc_id}.pdf").exists()
    assert client.get(f"/api/documents/{doc_id}").status_code == 404
    assert client.delete(f"/api/documents/{doc_id}").status_code == 404
    # the same file can be uploaded again after deletion
    assert upload(client, make_pdf(BIO), "bio.pdf").status_code == 202


def test_data_survives_restart(settings):
    app1 = create_app(settings=settings, embedder=FakeEmbedder(), ollama=FakeOllama())
    c1 = TestClient(app1)
    keep = upload(c1, make_pdf(BIO), "keep.pdf").json()["id"]
    drop = upload(c1, make_pdf(FIN), "drop.pdf").json()["id"]
    sid = c1.post("/api/chat", json={"question": "What do mitochondria produce?"}).json()["session_id"]
    c1.delete(f"/api/documents/{drop}")

    app2 = create_app(settings=settings, embedder=FakeEmbedder(), ollama=FakeOllama())  # "restart"
    c2 = TestClient(app2)
    docs = c2.get("/api/documents").json()["documents"]
    assert [d["id"] for d in docs] == [keep] and docs[0]["status"] == "ready"
    res = c2.post("/api/chat", json={"question": "What do mitochondria produce?", "document_ids": [keep]})
    assert res.json()["sources"][0]["filename"] == "keep.pdf"
    assert len(c2.get(f"/api/chat/sessions/{sid}").json()["messages"]) == 2


def test_interrupted_processing_marked_failed_on_restart(settings):
    app1 = create_app(settings=settings, embedder=FakeEmbedder(), ollama=FakeOllama())
    services = app1.state.services
    services.document_store.create("stuck1234567890a", "stuck.pdf", "stuck1234567890a.pdf", 10)
    app2 = create_app(settings=settings, embedder=FakeEmbedder(), ollama=FakeOllama())
    doc = app2.state.services.document_store.get("stuck1234567890a")
    assert doc["status"] == "failed"


def test_cors_allows_vite_origin(client):
    res = client.options(
        "/api/health",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
    )
    assert res.headers["access-control-allow-origin"] == "http://localhost:5173"
