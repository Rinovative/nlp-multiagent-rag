"""Deterministic Streamlit application-boundary tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import dotenv


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
    "MAX_UPLOAD_MB",
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
    assert app.info[0].value.startswith("PDF indexing is local.")


def test_configured_huggingface_script_starts_without_inference(monkeypatch):
    _isolate_configuration(monkeypatch)
    monkeypatch.setenv("HUGGINGFACE_API_TOKEN", "test-placeholder-token")
    monkeypatch.setenv("GENERATION_PROVIDER", "huggingface")

    app = AppTest.from_file("app.py").run(timeout=20)

    assert not app.exception
    assert not app.error
    assert not app.info
