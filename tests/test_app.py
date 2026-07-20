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
    assert app.sidebar.title[0].value == "Multilingual PDF RAG"
    assert not app.main.title
    assert app.sidebar.info[0].value.startswith("Die PDF-Indexierung erfolgt lokal.")
    assert app.main.info[0].value == (
        "Lade zuerst PDF-Dokumente über die Seitenleiste hoch."
    )
    assert app.chat_input[0].disabled is True
    assert not app.sidebar.chat_input


def test_configured_huggingface_script_starts_without_inference(monkeypatch):
    _isolate_configuration(monkeypatch)
    monkeypatch.setenv("HUGGINGFACE_API_TOKEN", "test-placeholder-token")
    monkeypatch.setenv("GENERATION_PROVIDER", "huggingface")

    app = AppTest.from_file("app.py").run(timeout=20)

    assert not app.exception
    assert not app.error
    assert not app.sidebar.info
    assert app.main.info[0].value == (
        "Lade zuerst PDF-Dokumente über die Seitenleiste hoch."
    )


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
    assert {
        "title",
        "write",
        "file_uploader",
        "caption",
        "spinner",
        "error",
        "success",
        "info",
    } <= sidebar_attributes
    assert {"chat_message", "chat_input"}.isdisjoint(sidebar_attributes)
    assert {"form", "text_input", "form_submit_button"}.isdisjoint(
        all_streamlit_attributes
    )
    assert "chat_input" in all_streamlit_attributes


class _FakeApplicationSession:
    def __init__(self) -> None:
        self.active_document_count = 1
        self.vector_store = SimpleNamespace(record_count=2)
        self.ask_calls: list[str] = []

    def sync_uploads(self, _uploads):
        return SimpleNamespace(changed=False, processed=(), active_document_ids=("a",))

    def ask(self, question):
        self.ask_calls.append(question)
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
        error=application.session.UploadValidationError("Upload-Limit überschritten.")
    )
    validation_app = _run_with_session(monkeypatch, validation_session)

    assert not validation_app.exception
    assert validation_app.sidebar.error[0].value == "Upload-Limit überschritten."
    assert not validation_app.main.error

    processing_session = _ReportingApplicationSession()
    processing_app = _run_with_session(monkeypatch, processing_session)

    assert not processing_app.exception
    assert processing_app.sidebar.success[0].value == (
        "1 PDF-Dokument(e) wurden erfolgreich als 3 durchsuchbare Chunks indexiert."
    )
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
        caption.value == "Aktiv: 1 Dokument(e), 2 Chunks."
        for caption in app.sidebar.caption
    )
    assert any(
        caption.value == "Antwortanbieter: huggingface (Qwen/Qwen2.5-7B-Instruct)"
        for caption in app.main.caption
    )
    assert len(app.main.chat_message) == 2
    assert any(expander.label == "Verwendete Quellen" for expander in app.main.expander)
    assert any(text.value == "• Projektbericht.pdf · Seite 1" for text in app.main.text)
    assert any(text.value == "• Anhang.pdf" for text in app.main.text)

    app.run(timeout=20)

    assert fake_session.ask_calls == ["An welcher Hochschule?"]
    assert len(app.session_state["chat_messages"]) == 2
