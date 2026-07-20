"""Validate runtime package boundaries and import safety."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import src


EXPECTED_DOMAINS = {
    "agents",
    "application",
    "cli",
    "configuration",
    "embeddings",
    "ingestion",
    "memory",
    "orchestration",
    "providers",
    "quota",
    "vectorstore",
}


def test_public_package_exports_resolve():
    assert set(src.__all__) == EXPECTED_DOMAINS
    assert len(src.__all__) == len(set(src.__all__))

    for domain_name in src.__all__:
        domain = getattr(src, domain_name)
        exports = getattr(domain, "__all__")
        assert len(exports) == len(set(exports))
        assert all(hasattr(domain, name) for name in exports)


def test_core_modules_do_not_import_streamlit():
    for path in Path("src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_modules = {
            alias.name.partition(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported_modules.update(
            node.module.partition(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )
        assert "streamlit" not in imported_modules, path


def test_importing_src_is_side_effect_free_in_a_fresh_process(workspace_tmp_path):
    project_root = Path.cwd().resolve()
    script = textwrap.dedent(
        """
        from pathlib import Path
        import sys

        import faiss
        import huggingface_hub
        import openai
        import redis
        import sentence_transformers

        calls = []

        def unexpected_call(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("Package import constructed a runtime dependency")

        openai.OpenAI = unexpected_call
        huggingface_hub.InferenceClient = unexpected_call
        redis.from_url = unexpected_call
        sentence_transformers.SentenceTransformer = unexpected_call
        faiss.IndexFlatL2 = unexpected_call

        import src

        assert calls == []
        assert "src.cli.cli_quota" not in sys.modules
        assert list(Path.cwd().iterdir()) == []
        """
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(project_root)

    subprocess.run(
        [sys.executable, "-c", script],
        cwd=workspace_tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
