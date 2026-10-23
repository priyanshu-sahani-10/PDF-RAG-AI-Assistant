from app.services.vector_store import VectorStore
from langchain_core.documents import Document
from tests.conftest import FakeEmbedder


def chunk(doc_id, idx, text, page=1):
    return Document(
        page_content=text,
        metadata={"document_id": doc_id, "filename": f"{doc_id}.pdf", "page_number": page, "chunk_index": idx},
    )


def build(tmp_path):
    emb = FakeEmbedder()
    store = VectorStore(tmp_path / "chroma")
    chunks = [
        chunk("a", 0, "photosynthesis converts sunlight into chemical energy in plants"),
        chunk("a", 1, "mitochondria produce energy for the cell", page=2),
        chunk("b", 0, "the stock market closed higher on tuesday"),
    ]
    store.add_chunks(chunks, emb.embed_documents([c.page_content for c in chunks]))
    return store, emb


def test_cosine_relevance_semantics(tmp_path):
    store, emb = build(tmp_path)
    hits = store.search(emb.embed_query("photosynthesis converts sunlight into chemical energy in plants"), 3)
    assert hits[0].text.startswith("photosynthesis")
    assert hits[0].distance < 0.01  # identical text: cosine distance ~ 0
    assert hits[0].relevance > 0.99
    assert hits[-1].relevance < hits[0].relevance


def test_document_filter_applies_inside_search(tmp_path):
    store, emb = build(tmp_path)
    query = emb.embed_query("stock market")
    only_a = store.search(query, 5, ["a"])
    assert only_a and all(h.document_id == "a" for h in only_a)
    both = store.search(query, 5, ["a", "b"])
    assert {h.document_id for h in both} == {"a", "b"}
    assert store.search(query, 5, []) == []


def test_delete_document_removes_vectors(tmp_path):
    store, emb = build(tmp_path)
    store.delete_document("a")
    assert store.count_for_document("a") == 0
    assert store.count_for_document("b") == 1


def test_persists_on_disk(tmp_path):
    build(tmp_path)
    reopened = VectorStore(tmp_path / "chroma")
    assert reopened.count_for_document("a") == 2
