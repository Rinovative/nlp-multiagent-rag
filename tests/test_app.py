"""Deterministic Streamlit application-boundary tests."""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import dotenv

from src import application, orchestration, providers


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


def test_streamlit_script_starts_without_credentials(monkeypatch):
    _isolate_configuration(monkeypatch)

    app = AppTest.from_file("app.py").run(timeout=20)

    assert not app.exception
    assert not app.sidebar.title
    assert app.sidebar.markdown[0].value == "**Multilingual PDF RAG**"
    assert app.sidebar.subheader[0].value == "Documents"
    assert (
        sum(
            element.value == "Upload PDFs and ask questions in English or German."
            for element in app.sidebar.markdown
        )
        == 1
    )
    assert not app.main.title
    assert not app.sidebar.info
    assert app.main.caption[0].value == "Upload PDFs in the sidebar to start chatting."
    assert app.chat_input[0].disabled is True
    assert app.chat_input[0].placeholder == "Ask a question about your documents"
    assert not app.sidebar.chat_input


def test_configured_huggingface_script_starts_without_inference(monkeypatch):
    _isolate_configuration(monkeypatch)
    monkeypatch.setenv("HUGGINGFACE_API_TOKEN", "test-placeholder-token")
    monkeypatch.setenv("GENERATION_PROVIDER", "huggingface")

    app = AppTest.from_file("app.py").run(timeout=20)

    assert not app.exception
    assert not app.error
    assert not app.sidebar.info
    assert app.main.caption[0].value == "Upload PDFs in the sidebar to start chatting."


def test_streamlit_layout_keeps_documents_in_sidebar_and_chat_in_main():
    tree = ast.parse(Path("app.py").read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    sidebar = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Attribute)
            and isinstance(item.context_expr.value, ast.Name)
            and item.context_expr.value.id == "st"
            and item.context_expr.attr == "sidebar"
            for item in node.items
        )
    )
    sidebar_calls = [node for node in ast.walk(sidebar) if isinstance(node, ast.Call)]
    sidebar_attributes = {
        node.func.attr
        for node in sidebar_calls
        if isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "st"
    }
    uploader = next(
        node
        for node in sidebar_calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "file_uploader"
    )
    max_upload_size = next(
        keyword.value
        for keyword in uploader.keywords
        if keyword.arg == "max_upload_size"
    )
    uploader_label = uploader.args[0]
    uploader_help = next(
        keyword.value for keyword in uploader.keywords if keyword.arg == "help"
    )

    assert ast.unparse(max_upload_size) == "config.max_upload_file_mb"
    all_streamlit_attributes = {
        node.func.attr
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "st"
    }
    top_level_streamlit_calls = [
        statement.value.func.attr
        for statement in tree.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Attribute)
        and isinstance(statement.value.func.value, ast.Name)
        and statement.value.func.value.id == "st"
    ]

    assert top_level_streamlit_calls[0] == "set_page_config"
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
    assert ast.literal_eval(page_title) == "Multilingual PDF RAG"
    assert {
        "markdown",
        "subheader",
        "write",
        "file_uploader",
        "caption",
        "spinner",
        "error",
        "success",
        "info",
    } <= sidebar_attributes
    assert "title" not in sidebar_attributes
    assert ast.literal_eval(uploader_label) == "Upload PDFs"
    assert "Up to " in ast.unparse(uploader_help)
    assert " PDFs · " in ast.unparse(uploader_help)
    assert " MB per file · " in ast.unparse(uploader_help)
    assert " MB total" in ast.unparse(uploader_help)
    assert {"chat_message", "chat_input"}.isdisjoint(sidebar_attributes)
    assert {"form", "text_input", "form_submit_button"}.isdisjoint(
        all_streamlit_attributes
    )
    assert "chat_input" in all_streamlit_attributes


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


def _run_with_session(monkeypatch, application_session):
    _isolate_configuration(monkeypatch)
    app = AppTest.from_file("app.py")
    app.session_state["session_id"] = "test-session"
    app.session_state["application_session"] = application_session
    app.session_state["chat_messages"] = []
    return app.run(timeout=20)


def test_sidebar_reports_upload_validation_and_processing_without_reprocessing(
    monkeypatch,
):
    validation_session = _ReportingApplicationSession(
        error=application.session.UploadValidationError("Upload limit exceeded.")
    )
    validation_app = _run_with_session(monkeypatch, validation_session)

    assert not validation_app.exception
    assert validation_app.sidebar.error[0].value == "Upload limit exceeded."
    assert not validation_app.main.error

    processing_session = _ReportingApplicationSession()
    processing_app = _run_with_session(monkeypatch, processing_session)

    assert not processing_app.exception
    assert processing_app.sidebar.success[0].value == "1 document · 3 chunks indexed"
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
        caption.value == "1 document · 2 chunks indexed"
        for caption in app.sidebar.caption
    )
    assert any(
        caption.value == "Hugging Face · Qwen/Qwen2.5-7B-Instruct"
        for caption in app.main.caption
    )
    assert len(app.main.chat_message) == 2
    assert any(expander.label == "Sources" for expander in app.main.expander)
    assert any(text.value == "• Projektbericht.pdf · page 1" for text in app.main.text)
    assert any(text.value == "• Anhang.pdf" for text in app.main.text)

    app.run(timeout=20)

    assert fake_session.ask_calls == ["An welcher Hochschule?"]
    assert len(app.session_state["chat_messages"]) == 2


def test_failed_free_fallback_has_specific_safe_visitor_message(monkeypatch):
    fallback_error = providers.contracts.GenerationFallbackError(
        provider_id="huggingface",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        fallback_reason="session_requests_exhausted",
        provider_error=providers.contracts.GenerationTemporaryError("private detail"),
    )
    fake_session = _FakeApplicationSession(error=fallback_error)
    app = _run_with_session(monkeypatch, fake_session)

    app.chat_input[0].set_value("What is this about?").run(timeout=20)

    assert fake_session.ask_calls == ["What is this about?"]
    assert app.main.error[0].value == (
        "The free fallback provider is temporarily unavailable. "
        "Please try again later."
    )
    assert "private detail" not in app.main.error[0].value
