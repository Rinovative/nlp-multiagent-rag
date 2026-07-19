import hashlib
import importlib

import huggingface_hub
import openai
import pytest
import redis
import sentence_transformers

from src import application, configuration, ingestion, memory, vectorstore

AppConfig = configuration.runtime.AppConfig
ConfigurationError = configuration.runtime.ConfigurationError
DocumentProcessingError = ingestion.processor.DocumentProcessingError
PreparedDocument = ingestion.processor.PreparedDocument
ProcessingResult = ingestion.processor.ProcessingResult
InMemoryConversationStore = memory.in_memory.InMemoryConversationStore
SessionDocumentManager = application.session.SessionDocumentManager
UploadedDocument = application.session.UploadedDocument
UploadValidationError = application.session.UploadValidationError
FAISSStore = vectorstore.faiss.FAISSStore


class FakeProcessor:
    def __init__(self, _store, calls):
        self.calls = calls

    def prepare_bytes(self, content, *, file_name):
        document_id = hashlib.sha256(content).hexdigest()
        self.calls.append(document_id)
        if content == b"FAIL":
            raise DocumentProcessingError("simulated processing failure")
        vector = [1.0, 0.0] if content.startswith(b"A") else [0.0, 1.0]
        embedded_chunk = {
            "chunk_id": f"{document_id}:000000:paragraph:part-0000",
            "text": content.decode("ascii"),
            "metadata": {
                "schema_version": 1,
                "length_unit": "characters",
                "document_id": document_id,
                "document_title": file_name,
                "source_type": "paragraph",
                "source_sequence": 0,
                "chunk_sequence": 0,
                "part_index": 0,
                "part_count": 1,
                "page_number": 1,
            },
            "embedding": vector,
        }
        return PreparedDocument(
            result=ProcessingResult(document_id, file_name, 1),
            embedded_chunks=(embedded_chunk,),
        )


def manager(calls):
    def store_factory():
        return FAISSStore(dimension=2, embedding_model="fake")

    return SessionDocumentManager(
        store_factory=store_factory,
        processor_factory=lambda store: FakeProcessor(store, calls),
        max_upload_bytes=1024,
    )


def test_two_sessions_cannot_retrieve_each_others_uploads():
    first = manager([])
    second = manager([])
    first.sync([UploadedDocument("shared.pdf", b"A document")])
    second.sync([UploadedDocument("shared.pdf", b"B document")])

    first_results = first.store.search([1.0, 0.0], k=10)
    second_results = second.store.search([0.0, 1.0], k=10)

    assert [result["text"] for result in first_results] == ["A document"]
    assert [result["text"] for result in second_results] == ["B document"]
    assert (
        first_results[0]["metadata"]["document_id"]
        != second_results[0]["metadata"]["document_id"]
    )


def test_same_filename_different_content_does_not_collide():
    session = manager([])
    result = session.sync(
        [
            UploadedDocument("same.pdf", b"A content"),
            UploadedDocument("same.pdf", b"B content"),
        ]
    )

    assert result.changed is True
    assert session.store.record_count == 2
    assert len({record["chunk_id"] for record in session.store.records}) == 2


def test_upload_changes_are_detected_without_reembedding_unchanged_documents():
    calls = []
    session = manager(calls)
    first = UploadedDocument("one.pdf", b"A content")
    second = UploadedDocument("two.pdf", b"B content")

    initial = session.sync([first])
    unchanged = session.sync([first])
    expanded = session.sync([first, second])
    reduced = session.sync([second])

    assert initial.changed is True
    assert unchanged.changed is False
    assert expanded.changed is True
    assert reduced.changed is True
    assert calls == [first.content_hash, second.content_hash]
    assert session.store.record_count == 1
    assert session.store.records[0]["metadata"]["document_id"] == second.content_hash


def test_conversation_history_is_explicitly_session_specific():
    store = InMemoryConversationStore(max_history=10)
    store.append("session-a", "user", "question a")
    store.append("session-b", "user", "question b")

    assert store.get_history("session-a") == [{"role": "user", "content": "question a"}]
    assert store.get_history("session-b") == [{"role": "user", "content": "question b"}]


def test_failed_upload_change_preserves_previous_session_store():
    session = manager([])
    session.sync([UploadedDocument("good.pdf", b"A good document")])

    with pytest.raises(DocumentProcessingError, match="simulated processing failure"):
        session.sync([UploadedDocument("bad.pdf", b"FAIL")])

    assert session.store.record_count == 1
    assert session.store.records[0]["text"] == "A good document"


def test_oversized_upload_is_rejected_before_processing():
    calls = []
    session = manager(calls)

    with pytest.raises(UploadValidationError, match="upload limit"):
        session.sync([UploadedDocument("large.pdf", b"x" * 1025)])

    assert calls == []
    assert session.store.record_count == 0


def test_combined_upload_limit_is_checked_before_processing():
    calls = []
    session = manager(calls)

    with pytest.raises(UploadValidationError, match="combined upload limit"):
        session.sync(
            [
                UploadedDocument("one.pdf", b"A" * 600),
                UploadedDocument("two.pdf", b"B" * 600),
            ]
        )

    assert calls == []
    assert session.store.record_count == 0


def test_importing_application_factory_has_no_external_or_file_side_effects(
    workspace_tmp_path, monkeypatch
):
    monkeypatch.chdir(workspace_tmp_path)

    def unexpected_call(*_args, **_kwargs):
        raise AssertionError("Import attempted to create an external client")

    monkeypatch.setattr(openai, "OpenAI", unexpected_call)
    monkeypatch.setattr(huggingface_hub, "InferenceClient", unexpected_call)
    monkeypatch.setattr(redis, "from_url", unexpected_call)
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", unexpected_call)

    import src.application.application_factory as factory_module

    importlib.reload(factory_module)
    assert list(workspace_tmp_path.iterdir()) == []


def test_local_session_starts_without_optional_credentials(
    workspace_tmp_path, monkeypatch
):
    monkeypatch.chdir(workspace_tmp_path)
    current_session = application.factory.create_application_session(
        session_id="test-session", config=AppConfig()
    )

    assert current_session.session_id == "test-session"
    assert current_session.vector_store.record_count == 0
    assert list(workspace_tmp_path.iterdir()) == []
    with pytest.raises(ConfigurationError, match="HUGGINGFACE_API_TOKEN"):
        current_session.ask("Can this run without a configured provider?")


def test_embedding_provider_is_reused_across_application_sessions():
    config = AppConfig()

    first = application.factory.create_embedding_provider(config)
    second = application.factory.create_embedding_provider(config)

    assert first is second
