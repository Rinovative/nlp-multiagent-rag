"""Deterministic Streamlit application-boundary tests."""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import dotenv
import pytest

from src import (
    agents,
    application,
    ingestion,
    memory,
    orchestration,
    providers,
    quota,
    vectorstore,
)


class _WorkspaceTemporaryDirectory:
    def __init__(self) -> None:
        path = Path(".pytest-tmp/streamlit-app-test").resolve()
        path.mkdir(parents=True, exist_ok=True)
        self.name = str(path)


with patch.object(tempfile, "TemporaryDirectory", _WorkspaceTemporaryDirectory):
    from streamlit.testing.v1 import AppTest


_CONFIGURATION_NAMES = {
    "HUGGINGFACE_API_TOKEN",
    "HUGGINGFACE_GENERATION_MODEL",
    "GENERATION_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
    "EMBEDDING_BATCH_SIZE",
    "OPENAI_API_KEY",
    "OPENAI_GENERATION_MODEL",
    "OPENAI_FALLBACK_ENABLED",
    "REDIS_URL",
    "OPENAI_QUOTA_KEY_PREFIX",
    "MAX_UPLOAD_FILE_MB",
    "MAX_UPLOAD_TOTAL_MB",
    "MAX_UPLOAD_FILES",
    "MAX_INPUT_CHARACTERS",
    "MAX_OUTPUT_TOKENS",
    "MAX_HISTORY_MESSAGES",
    "RETRIEVAL_TOP_K",
    "PROVIDER_TIMEOUT_SECONDS",
}


def _isolate_configuration(monkeypatch) -> None:
    for name in _CONFIGURATION_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *_args, **_kwargs: False)


def test_product_identity_layout_and_credential_free_startup(monkeypatch):
    _isolate_configuration(monkeypatch)

    app = AppTest.from_file("app.py").run(timeout=20)

    assert not app.exception
    assert not app.sidebar.title
    assert app.sidebar.markdown[0].value == "**PDF RAG Assistant**"
    assert len(app.sidebar.file_uploader) == 1
    assert not app.main.file_uploader
    assert not app.main.title
    assert app.chat_input[0].disabled is True
    assert not app.sidebar.chat_input

    tree = ast.parse(Path("app.py").read_text(encoding="utf-8"))
    page_config = next(
        statement.value
        for statement in tree.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Attribute)
        and statement.value.func.attr == "set_page_config"
    )
    page_title = next(
        keyword.value for keyword in page_config.keywords if keyword.arg == "page_title"
    )
    assert ast.literal_eval(page_title) == "PDF RAG Assistant"


class _FakeApplicationSession:
    def __init__(self, *, error=None) -> None:
        self.active_document_count = 1
        self.vector_store = SimpleNamespace(record_count=2)
        self.ask_calls: list[str] = []
        self.error = error

    def sync_uploads(self, _uploads):
        return SimpleNamespace(changed=False, processed=(), active_document_ids=("a",))

    def ask(self, question):
        self.ask_calls.append(question)
        if self.error is not None:
            raise self.error
        return orchestration.rag.RAGResult(
            generation=providers.contracts.GenerationResult(
                answer="Die OST – Ostschweizer Fachhochschule.",
                provider_id="huggingface",
                model_id="Qwen/Qwen2.5-7B-Instruct",
                usage=providers.contracts.GenerationUsage(20, 8),
            ),
            sources=(
                orchestration.rag.SourceReference("Projektbericht.pdf", 1),
                orchestration.rag.SourceReference("Anhang.pdf"),
            ),
        )


class _ReportingApplicationSession:
    def __init__(self, *, error=None) -> None:
        self.active_document_count = 1
        self.vector_store = SimpleNamespace(record_count=3)
        self.error = error
        self.sync_calls = 0
        self.processing_runs = 0

    def sync_uploads(self, _uploads):
        self.sync_calls += 1
        if self.error is not None:
            raise self.error
        if self.processing_runs == 0:
            self.processing_runs += 1
            return SimpleNamespace(
                changed=True,
                processed=(SimpleNamespace(chunk_count=3),),
                active_document_ids=("a",),
            )
        return SimpleNamespace(
            changed=False,
            processed=(),
            active_document_ids=("a",),
        )


class _DeterministicEmbeddingProvider:
    model_id = "test-embedding"
    dimension = 2

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _text in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def _minimal_pdf(text):
    content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii")
    objects = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        b"<< /Length "
        + str(len(content)).encode("ascii")
        + b" >>\nstream\n"
        + content
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    )
    payload = bytearray(b"%PDF-1.4\n")
    offsets = []
    for object_number, pdf_object in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload.extend(f"{object_number} 0 obj\n".encode("ascii"))
        payload.extend(pdf_object)
        payload.extend(b"\nendobj\n")
    xref_offset = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets:
        payload.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    payload.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(payload)


class _GenerationProvider:
    def __init__(self, provider_id, *, error=None) -> None:
        self.provider_id = provider_id
        self.model_id = f"{provider_id}-model"
        self.error = error
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return providers.contracts.GenerationResult(
            answer=f"{self.provider_id} answer",
            provider_id=self.provider_id,
            model_id=self.model_id,
            usage=providers.contracts.GenerationUsage(4, 3),
        )


class _SessionQuotaExhausted:
    def __init__(self) -> None:
        self.reserve_calls = 0

    def reserve(self, **_kwargs):
        self.reserve_calls += 1
        raise quota.contracts.QuotaExhaustedError("session_requests_exhausted")

    def inspect(self, *, now=None):
        raise AssertionError("inspect is not used by routing")

    def set_limits(self, limits):
        raise AssertionError("set_limits is not used by routing")

    def set_enabled(self, enabled):
        raise AssertionError("set_enabled is not used by routing")

    def reconcile(self, reservation, *, actual_tokens):
        raise AssertionError("reconcile is unreachable after a denied reservation")

    def release(self, reservation):
        raise AssertionError("release is unreachable after a denied reservation")


def _production_failure_session():
    embedding_provider = _DeterministicEmbeddingProvider()
    free_provider = _GenerationProvider(
        "huggingface",
        error=providers.contracts.GenerationAuthenticationError(
            "private hosted credential detail"
        ),
    )
    openai_provider = _GenerationProvider("openai")
    quota_backend = _SessionQuotaExhausted()
    router = providers.router.GenerationRouter(
        mode="auto",
        free_provider=free_provider,
        openai_provider=openai_provider,
        quota_backend=quota_backend,
    )

    def store_factory():
        return vectorstore.faiss.FAISSStore(
            dimension=embedding_provider.dimension,
            embedding_model=embedding_provider.model_id,
        )

    document_manager = application.session.SessionDocumentManager(
        store_factory=store_factory,
        processor_factory=lambda store: ingestion.processor.DocumentProcessor(
            faiss_store=store,
            embedding_provider=embedding_provider,
        ),
        max_upload_file_bytes=4096,
        max_upload_total_bytes=4096,
        max_upload_files=10,
    )
    conversation_store = memory.in_memory.InMemoryConversationStore(max_history=10)
    conversation_store.append("test-session", "user", "Earlier question")
    conversation_store.append("test-session", "assistant", "Earlier answer")

    def chatbot_factory(store, session_conversation_store):
        return orchestration.rag.RAGChatbot(
            retriever_agent=agents.retriever.RetrieverAgent(
                store, embedding_provider, top_k=2
            ),
            generator_agent=agents.generator.GeneratorAgent(router),
            memory_agent=agents.memory.MemoryAgent(session_conversation_store),
        )

    application_session = application.session.ApplicationSession(
        session_id="test-session",
        document_manager=document_manager,
        conversation_store=conversation_store,
        chatbot_factory=chatbot_factory,
    )
    return (
        application_session,
        conversation_store,
        free_provider,
        openai_provider,
        quota_backend,
    )


def _run_with_session(monkeypatch, application_session, *, chat_messages=None):
    _isolate_configuration(monkeypatch)
    app = AppTest.from_file("app.py")
    app.session_state["session_id"] = "test-session"
    app.session_state["application_session"] = application_session
    app.session_state["chat_messages"] = list(chat_messages or [])
    return app.run(timeout=20)


def test_sidebar_reports_upload_validation_and_processing_without_reprocessing(
    monkeypatch,
):
    validation_session = _ReportingApplicationSession(
        error=application.session.UploadValidationError("Upload limit exceeded.")
    )
    validation_app = _run_with_session(monkeypatch, validation_session)

    assert not validation_app.exception
    assert len(validation_app.sidebar.error) == 1
    assert not validation_app.main.error

    processing_session = _ReportingApplicationSession()
    processing_app = _run_with_session(monkeypatch, processing_session)

    assert not processing_app.exception
    assert len(processing_app.sidebar.success) == 1
    assert not processing_app.main.success

    processing_app.run(timeout=20)

    assert processing_session.sync_calls == 2
    assert processing_session.processing_runs == 1
    assert not processing_app.sidebar.success


def test_chat_submission_persists_once_with_attribution_and_sources(monkeypatch):
    fake_session = _FakeApplicationSession()
    app = _run_with_session(monkeypatch, fake_session)

    assert app.chat_input[0].disabled is False
    assert not app.sidebar.chat_input
    app.chat_input[0].set_value("An welcher Hochschule?").run(timeout=20)

    assert fake_session.ask_calls == ["An welcher Hochschule?"]
    assert len(app.session_state["chat_messages"]) == 2
    assert any(
        "Hugging Face" in caption.value and "Qwen/Qwen2.5-7B-Instruct" in caption.value
        for caption in app.main.caption
    )
    assert len(app.main.chat_message) == 2
    assert any(expander.label == "Sources" for expander in app.main.expander)
    source_labels = [text.value for text in app.main.text]
    assert "Projektbericht.pdf" in source_labels[0]
    assert "page 1" in source_labels[0]
    assert "Anhang.pdf" in source_labels[1]

    app.run(timeout=20)

    assert fake_session.ask_calls == ["An welcher Hochschule?"]
    assert len(app.session_state["chat_messages"]) == 2


_REPRESENTATIVE_GENERATION_ERRORS = (
    providers.contracts.GenerationConfigurationError("private detail"),
    providers.contracts.GenerationCreditsError("private detail"),
    providers.contracts.GenerationModelUnavailableError("private detail"),
    providers.contracts.GenerationInvalidRequestError("private detail"),
    providers.contracts.GenerationSafetyError("private detail"),
)


@pytest.mark.parametrize(
    "generation_error",
    _REPRESENTATIVE_GENERATION_ERRORS,
    ids=lambda error: type(error).__name__,
)
def test_generation_errors_render_safely_without_fabricated_output(
    monkeypatch, generation_error
):
    fake_session = _FakeApplicationSession(error=generation_error)
    app = _run_with_session(monkeypatch, fake_session)

    app.chat_input[0].set_value("What is this about?").run(timeout=20)

    assert not app.exception
    assert fake_session.ask_calls == ["What is this about?"]
    assert len(app.main.error) == 1
    assert "private detail" not in app.main.error[0].value
    assert len(app.session_state["chat_messages"]) == 1
    assert not app.main.caption
    assert not app.main.expander


def test_quota_to_huggingface_authentication_failure_stays_inside_streamlit_boundary(
    monkeypatch,
):
    (
        application_session,
        conversation_store,
        free_provider,
        openai_provider,
        quota_backend,
    ) = _production_failure_session()
    previous_chat = [
        {"role": "user", "content": "Earlier question"},
        {
            "role": "assistant",
            "content": "Earlier answer",
            "provider_id": "openai",
            "model_id": "previous-model",
            "fallback_occurred": False,
            "sources": [{"document_name": "Previous.pdf", "page_number": 1}],
        },
    ]
    app = _run_with_session(
        monkeypatch,
        application_session,
        chat_messages=previous_chat,
    )
    app.sidebar.file_uploader[0].set_value(
        (
            "Indexed.pdf",
            _minimal_pdf("Previously indexed document context."),
            "application/pdf",
        )
    ).run(timeout=20)

    assert application_session.active_document_count == 1
    assert application_session.vector_store.record_count == 1
    assert app.chat_input[0].disabled is False

    app.chat_input[0].set_value("Failed fallback question").run(timeout=20)

    assert not app.exception
    assert quota_backend.reserve_calls == 1
    assert openai_provider.calls == 0
    assert free_provider.calls == 1
    assert len(app.main.error) == 1
    assert "private hosted credential detail" not in app.main.error[0].value
    assert list(app.session_state["chat_messages"]) == [
        *previous_chat,
        {"role": "user", "content": "Failed fallback question"},
    ]
    assert conversation_store.get_history("test-session") == [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
    ]
    assert application_session.active_document_count == 1
    assert application_session.vector_store.record_count == 1
    assert (
        sum(
            element.value == "Failed fallback question" for element in app.main.markdown
        )
        == 1
    )
    assert len(app.main.chat_message) == 4
    assert all("Hugging Face" not in caption.value for caption in app.main.caption)
    assert len(app.main.expander) == 1

    free_provider.error = None
    app.chat_input[0].set_value("Retry question").run(timeout=20)

    assert not app.exception
    assert not app.main.error
    assert quota_backend.reserve_calls == 2
    assert openai_provider.calls == 0
    assert free_provider.calls == 2
    assert conversation_store.get_history("test-session") == [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": "Retry question"},
        {"role": "assistant", "content": "huggingface answer"},
    ]
    assert any(
        "Hugging Face" in caption.value and caption.value.endswith("huggingface-model")
        for caption in app.main.caption
    )
