from types import SimpleNamespace

import pytest

from freezer_rag.config import Settings
from freezer_rag.ingestion.corpus import SOURCES
from freezer_rag.models import Chunk, RagError, SourceUnit
from freezer_rag.retrieval.hybrid import Retriever
from freezer_rag.vectorstore.store import IndexStore, store_owner


@pytest.fixture
def index(tmp_path, monkeypatch):
    documents = tmp_path / "docs"
    documents.mkdir()
    for name in SOURCES:
        (documents / name).write_text("fixture content", encoding="utf-8")
    settings = Settings(
        _env_file=None,
        documents_dir=documents,
        qdrant_path=tmp_path / "indexes",
        manifests_dir=tmp_path / "manifests",
        processed_dir=tmp_path / "processed",
        final_top_k=1,
    )
    units = [
        SourceUnit(
            source=SOURCES[0],
            source_hash="fixture",
            title="Requirements",
            locator=f"page {index}",
            page=index,
            text=text,
        )
        for index, text in enumerate(
            [
                "FR-28 Rules require an active reservation before physical check-in.",
                "FR-29 Administrators maintain freezer restrictions and structure.",
            ],
            1,
        )
    ]
    monkeypatch.setattr("freezer_rag.vectorstore.store.load_corpus", lambda *args: units)
    embeddings = SimpleNamespace(
        settings=settings,
        digest="digest1",
        strategy="nomic-search-test",
        input_limit=2048,
        calls=0,
        fail=False,
    )

    def embed(texts, **kwargs):
        embeddings.calls += 1
        if embeddings.fail:
            raise RagError("Simulated embedding failure")
        return [[1.0] + [0.0] * 767 for text in texts]

    embeddings.embed = embed
    store = IndexStore(settings, embeddings)
    yield store, embeddings, units
    store.close()


def test_atomic_rebuild_persistence_no_duplicate_growth_and_stale_index_rejection(index):
    store, embeddings, units = index
    first = store.build(progress=lambda text: None)
    store.settings.llm_model = "another-chat-model:4b"
    assert store.read_manifest() == first
    calls = embeddings.calls
    assert store.build(progress=lambda text: None) == first
    assert embeddings.calls == calls
    embeddings.fail = True
    with pytest.raises(RagError, match="Simulated"):
        store.build(rebuild=True, progress=lambda text: None)
    assert store.read_manifest() == first
    embeddings.fail = False
    store.open()
    assert len(store.chunks) == first["chunk_count"]
    store.close()
    units.pop()
    replacement = store.build(rebuild=True, progress=lambda text: None)
    assert replacement["chunk_count"] == 1
    store.open()
    assert len(store.chunks) == 1
    store.close()
    embeddings.digest = "different-model-digest"
    with pytest.raises(RagError, match="incompatible"):
        store.open()


def test_hybrid_retrieval_promotes_exact_identifier_and_store_rejects_second_owner(index):
    store, embeddings, _ = index
    store.build(progress=lambda text: None)
    store.open()
    result = Retriever(store).search("FR-29")
    assert "FR-29" in result[0].text
    with store_owner(store.settings.qdrant_path):
        with pytest.raises(RagError, match="already in use"):
            with store_owner(store.settings.qdrant_path):
                pass


def test_restore_recovers_a_damaged_pointer_and_refuses_incompatible_snapshot(index):
    store, embeddings, _ = index
    original = store.build(progress=lambda text: None)
    store.build(rebuild=True, progress=lambda text: None)
    assert len(store.snapshots()) == 2
    store.pointer.write_text('{"broken":true}', encoding="utf-8")
    assert store.restore(original["snapshot"]) == original
    assert store.read_manifest() == original
    before = store.pointer.read_bytes()
    embeddings.digest = "changed-digest"
    with pytest.raises(RagError, match="incompatible"):
        store.restore(original["snapshot"])
    assert store.pointer.read_bytes() == before


def test_comparison_retrieval_keeps_both_documents_even_when_their_passages_match():
    texts = [
        ("Software_Design_Document.pdf", "POST /api/v1/conversations Agent conversation endpoint."),
        ("openapi.yaml", "POST /api/v1/conversations Agent conversation endpoint."),
        ("business_requirements.pdf", "Freezer temperature and storage capacity."),
        ("business_requirements.pdf", "Project membership and laboratory permissions."),
        ("business_requirements.pdf", "Materials and samples require appropriate handling."),
        (
            "openapi.yaml",
            "GET /api/v1/conversations/{conversation_id} Individual conversation details.",
        ),
    ]
    chunks = [
        Chunk(
            id=str(index),
            source=source,
            source_hash="test",
            title="Test",
            locator="test",
            text=text,
            normalized_text=text,
        )
        for index, (source, text) in enumerate(texts)
    ]
    settings = Settings(_env_file=None, final_top_k=2, lexical_top_k=1, dense_top_k=1)
    store = SimpleNamespace(
        chunks=chunks,
        settings=settings,
        embeddings=SimpleNamespace(embed=lambda *args, **kwargs: [[1.0]]),
        dense=lambda *args: ["2"],
    )
    required = ("Software_Design_Document.pdf", "openapi.yaml")
    found = Retriever(store).search("Agent conversations endpoint", required_sources=required)
    assert {chunk.source for chunk in found} == set(required)
    assert {chunk.id for chunk in found} == {"0", "1"}
