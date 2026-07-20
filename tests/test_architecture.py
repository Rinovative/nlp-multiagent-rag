"""Validate package boundaries, documentation contracts, and import safety."""

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import src


EXPECTED_SURFACES = {
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

_BANNER_SEPARATOR = "=" * 79
_MANDATORY_BANNER_SECTIONS = (
    "Responsibilities:",
    "Design principles:",
    "Boundaries:",
)


def _syntax_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _literal_all(tree: ast.Module) -> list[str] | None:
    for node in tree.body:
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
        ):
            value = node.value
        if isinstance(value, (ast.List, ast.Tuple)) and all(
            isinstance(element, ast.Constant) and isinstance(element.value, str)
            for element in value.elts
        ):
            return [element.value for element in value.elts]
    return None


def _relative_aliases(tree: ast.Module) -> list[str]:
    aliases: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.level != 1:
            continue
        aliases.extend(alias.asname or alias.name for alias in node.names)
    return aliases


def _documented_aliases(docstring: str, heading: str) -> list[str]:
    lines = docstring.splitlines()
    heading_index = lines.index(heading)
    aliases: list[str] = []
    for line in lines[heading_index + 1 :]:
        if not line.startswith("- "):
            continue
        alias, separator, description = line[2:].partition(":")
        assert separator and alias and description.strip()
        aliases.append(alias)
    return aliases


def _assert_section_entries(lines: list[str], heading: str, end: int) -> None:
    start = lines.index(heading) + 1
    entries = lines[start:end]
    assert entries and all(not entry or entry.startswith("  - ") for entry in entries)
    assert any(entry.startswith("  - ") and entry[4:].strip() for entry in entries)


def test_root_public_surface_matches_documented_domains():
    assert set(src.__all__) == EXPECTED_SURFACES
    assert all(hasattr(src, name) for name in src.__all__)


def test_substantive_modules_use_exact_banners_and_import_order():
    modules = [
        Path("app.py"),
        *(path for path in Path("src").rglob("*.py") if path.name != "__init__.py"),
    ]
    assert modules
    for path in modules:
        tree = _syntax_tree(path)
        docstring = ast.get_docstring(tree)
        assert docstring is not None
        lines = docstring.splitlines()
        assert lines[0] == _BANNER_SEPARATOR
        assert lines[1] == path.name
        assert lines[2] == _BANNER_SEPARATOR
        assert lines[-1] == _BANNER_SEPARATOR
        assert "..." not in docstring and "…" not in docstring

        section_indexes = [
            lines.index(section) for section in _MANDATORY_BANNER_SECTIONS
        ]
        assert section_indexes == sorted(section_indexes)
        assert section_indexes[0] > 3 and lines[section_indexes[0] - 1] == ""
        for index in section_indexes[1:]:
            assert lines[index - 1] == ""
        optional_notes = lines.index("Notes:") if "Notes:" in lines else len(lines) - 1
        assert optional_notes > section_indexes[-1]
        section_ends = [section_indexes[1], section_indexes[2], optional_notes]
        for heading, end in zip(_MANDATORY_BANNER_SECTIONS, section_ends, strict=True):
            _assert_section_entries(lines, heading, end)
        if "Notes:" in lines:
            _assert_section_entries(lines, "Notes:", len(lines) - 1)

        assert isinstance(tree.body[0], ast.Expr)
        future_import = tree.body[1]
        assert isinstance(future_import, ast.ImportFrom)
        assert future_import.module == "__future__"
        assert [alias.name for alias in future_import.names] == ["annotations"]


def test_every_package_documents_exact_public_surface():
    roots = [Path("src"), Path("tests")]
    init_files = [path for root in roots for path in root.rglob("__init__.py")]
    assert init_files
    for init_file in init_files:
        tree = _syntax_tree(init_file)
        docstring = ast.get_docstring(tree, clean=False)
        assert docstring is not None
        assert isinstance(tree.body[0], ast.Expr)
        future_import = tree.body[1]
        assert isinstance(future_import, ast.ImportFrom)
        assert future_import.module == "__future__"
        assert [alias.name for alias in future_import.names] == ["annotations"]
        declared = _literal_all(tree)
        assert declared is not None

        if init_file == Path("src/cli/__init__.py"):
            assert "Executable modules:" in docstring
            assert _documented_aliases(docstring, "Executable modules:") == [
                "cli_quota"
            ]
            assert _relative_aliases(tree) == []
            assert declared == []
            continue

        assert "Provides:" in docstring
        documented = _documented_aliases(docstring, "Provides:")
        imported = _relative_aliases(tree)
        assert documented == imported == declared

        module_name = ".".join(init_file.parent.parts)
        module = __import__(module_name, fromlist=["*"])
        assert all(hasattr(module, alias) for alias in declared)


def test_implementation_modules_declare_documented_public_apis():
    modules = [path for path in Path("src").rglob("*.py") if path.name != "__init__.py"]
    for path in modules:
        tree = _syntax_tree(path)
        declared = _literal_all(tree)
        assert declared is not None
        local_bindings: set[str] = set()
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                local_bindings.add(node.name)
            elif isinstance(node, ast.Assign):
                local_bindings.update(
                    target.id for target in node.targets if isinstance(target, ast.Name)
                )
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                local_bindings.add(node.target.id)
        assert set(declared) <= local_bindings
        definitions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for name in declared:
            definition = definitions.get(name)
            if definition is None:
                continue
            assert ast.get_docstring(definition)
            if isinstance(definition, ast.ClassDef):
                public_methods = [
                    node
                    for node in definition.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and not node.name.startswith("_")
                ]
                assert all(ast.get_docstring(method) for method in public_methods)


def test_importing_src_is_side_effect_free_in_a_fresh_process(workspace_tmp_path):
    project_root = Path.cwd().resolve()
    script = textwrap.dedent(
        """
        from pathlib import Path

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


def test_obsolete_generic_module_paths_are_absent():
    obsolete = [
        "src.config",
        "src.main",
        "src.pipeline",
        "src.session",
        "src.document_processor",
        "src.agents.generator_agent",
        "src.agents.summarizer_agent",
        "src.ingestion.embedder",
        "src.memory.memory",
        "src.utils.utils",
        "src.vectorstore.faiss_store",
    ]
    assert all(_module_path_is_absent(name) for name in obsolete)


def _module_path_is_absent(name: str) -> bool:
    """Return whether a module path is absent, including a missing parent package."""
    try:
        return importlib.util.find_spec(name) is None
    except ModuleNotFoundError:
        return True


def test_cli_package_remains_import_free():
    assert src.cli.__all__ == []


def test_unsupported_development_container_is_absent():
    """Reject tracked development-container files, not transient directories."""

    repository_root = Path(__file__).resolve().parents[1]
    tracked_paths = subprocess.run(
        ["git", "ls-files", "--", ".devcontainer"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    assert tracked_paths == []
