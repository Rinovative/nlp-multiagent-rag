"""Deterministic Streamlit application-boundary tests."""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import dotenv

from src import orchestration, providers


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
    assert app.title[0].value == "Multilingual PDF RAG"
    assert app.info[0].value.startswith("Die PDF-Indexierung erfolgt lokal.")
    assert app.chat_input[0].disabled is True


def test_configured_huggingface_script_starts_without_inference(monkeypatch):
    _isolate_configuration(monkeypatch)
    monkeypatch.setenv("HUGGINGFACE_API_TOKEN", "test-placeholder-token")
    monkeypatch.setenv("GENERATION_PROVIDER", "huggingface")

    app = AppTest.from_file("app.py").run(timeout=20)

    assert not app.exception
    assert not app.error
    assert not app.info


def test_streamlit_uploader_uses_per_file_limit_and_old_form_is_absent():
    tree = ast.parse(Path("app.py").read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    uploader = next(
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "file_uploader"
    )
    max_upload_size = next(
        keyword.value
        for keyword in uploader.keywords
        if keyword.arg == "max_upload_size"
    )

    assert ast.unparse(max_upload_size) == "config.max_upload_file_mb"
    called_attributes = {
        node.func.attr for node in calls if isinstance(node.func, ast.Attribute)
    }
    assert {"form", "text_input", "form_submit_button"}.isdisjoint(called_attributes)
    assert "chat_input" in called_attributes


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


def test_chat_submission_persists_once_with_attribution_and_sources(monkeypatch):
    _isolate_configuration(monkeypatch)
    fake_session = _FakeApplicationSession()
    app = AppTest.from_file("app.py")
    app.session_state["session_id"] = "test-session"
    app.session_state["application_session"] = fake_session
    app.session_state["chat_messages"] = []
    app.run(timeout=20)

    assert app.chat_input[0].disabled is False
    app.chat_input[0].set_value("An welcher Hochschule?").run(timeout=20)

    assert fake_session.ask_calls == ["An welcher Hochschule?"]
    assert len(app.session_state["chat_messages"]) == 2
    assert any(
        caption.value == "Aktiv: 1 Dokument(e), 2 Chunks." for caption in app.caption
    )
    assert any(
        caption.value == "Antwortanbieter: huggingface (Qwen/Qwen2.5-7B-Instruct)"
        for caption in app.caption
    )
    assert any(expander.label == "Verwendete Quellen" for expander in app.expander)
    assert any(text.value == "• Projektbericht.pdf · Seite 1" for text in app.text)
    assert any(text.value == "• Anhang.pdf" for text in app.text)

    app.run(timeout=20)

    assert fake_session.ask_calls == ["An welcher Hochschule?"]
    assert len(app.session_state["chat_messages"]) == 2
