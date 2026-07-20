"""Validate package boundaries, documentation contracts, and import safety."""

import ast
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap
from urllib.parse import unquote, urlparse

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
        if isinstance(value, (ast.List, ast.Tuple)):
            declared: list[str] = []
            for element in value.elts:
                if not isinstance(element, ast.Constant) or not isinstance(
                    element.value, str
                ):
                    break
                declared.append(element.value)
            else:
                return declared
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


def _readme_project_tree_entries() -> list[tuple[str, bool, str]]:
    readme_lines = Path("README.md").read_text(encoding="utf-8").splitlines()
    heading_index = readme_lines.index("## 📂 Projektstruktur")
    fence_start = readme_lines.index("```text", heading_index) + 1
    fence_end = readme_lines.index("```", fence_start)

    entries: list[tuple[str, bool, str]] = []
    directory_stack: list[str] = []
    for line in readme_lines[fence_start:fence_end]:
        branch_positions = [
            position
            for marker in ("├── ", "└── ")
            if (position := line.find(marker)) >= 0
        ]
        if not branch_positions:
            assert line == "."
            continue

        branch_position = min(branch_positions)
        prefix = line[:branch_position]
        assert len(prefix) % 4 == 0
        assert prefix.replace("│   ", "").replace("    ", "") == ""
        depth = len(prefix) // 4
        name_and_comment = line[branch_position + 4 :]
        name, _, comment = name_and_comment.partition("  # ")
        name = name.rstrip()
        assert name
        is_directory = name.endswith("/")
        basename = name.removesuffix("/")

        directory_stack = directory_stack[:depth]
        assert len(directory_stack) == depth
        path = "/".join([*directory_stack, basename])
        entries.append((path, is_directory, comment.strip()))
        if is_directory:
            directory_stack.append(basename)

    return entries


def _readme_section(readme: str, heading: str) -> str:
    """Return one top-level README section without depending on line spacing."""

    start = readme.index(heading) + len(heading)
    remainder = readme[start:]
    section_end = re.search(r"^---$", remainder, flags=re.MULTILINE)
    assert section_end is not None
    return remainder[: section_end.start()]


def _details_blocks(section: str) -> list[str]:
    """Return non-nested collapsed blocks from one README section."""

    return re.findall(r"<details>\s*(.*?)\s*</details>", section, flags=re.DOTALL)


def _details_summary(block: str) -> str:
    """Return the strong-text summary from one collapsed README block."""

    summary = re.search(r"<summary><strong>([^<]+)</strong></summary>", block)
    assert summary is not None
    return summary.group(1)


def test_root_public_surface_matches_documented_domains():
    assert set(src.__all__) == EXPECTED_SURFACES
    assert all(hasattr(src, name) for name in src.__all__)


def test_app_generation_error_references_exist_in_the_public_contract():
    tree = _syntax_tree(Path("app.py"))
    referenced_errors = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr.startswith("Generation")
        and node.attr.endswith("Error")
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "contracts"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "providers"
    }
    contracts = src.providers.contracts
    public_errors = {
        name
        for name in contracts.__all__
        if isinstance(getattr(contracts, name, None), type)
        and issubclass(getattr(contracts, name), contracts.GenerationError)
    }

    assert referenced_errors
    assert referenced_errors <= public_errors
    assert all(hasattr(contracts, name) for name in referenced_errors)


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


def test_readme_public_header_and_project_details_are_compact():
    readme = Path("README.md").read_text(encoding="utf-8")
    lines = readme.splitlines()

    assert lines[0] == (
        "[Interaktive Streamlit-Demo öffnen]"
        "(https://nlp-multiagent-rag.streamlit.app/)"
    )
    assert "# NLP Multi-Agent RAG" in lines
    assert "# NLP Multi-Agent RAG (Wahlfachprojekt)" not in lines
    assert any(line.startswith("**Projektart:** Wahlfachprojekt") for line in lines)

    project_section = _readme_section(readme, "## 📌 Projektbeschreibung")
    first_details = project_section.index("<details>")
    visible_paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", project_section[:first_details])
        if paragraph.strip()
    ]
    assert len(visible_paragraphs) == 1
    assert 70 <= len(visible_paragraphs[0].split()) <= 100

    details = _details_blocks(project_section)
    summaries = [_details_summary(block) for block in details]
    assert summaries == [
        "Dokumentverarbeitung und Retrieval",
        "Multi-Agent-Orchestrierung und Sitzungen",
        "Provider-Routing und Kontingentschutz",
        "Qualitätssicherung",
    ]
    assert "<details open" not in project_section


def test_readme_architecture_and_execution_use_the_intended_collapsed_structure():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert readme.count("<details>") == readme.count("</details>")

    architecture = _readme_section(readme, "## 🧭 Architektur und Datenfluss")
    diagram_blocks = _details_blocks(architecture)
    assert len(diagram_blocks) == 2
    assert [_details_summary(block) for block in diagram_blocks] == [
        "Dokumentaufnahme anzeigen",
        "Fragebeantwortung anzeigen",
    ]
    assert readme.count("```mermaid") == 2
    for block in diagram_blocks:
        diagrams = re.findall(r"```mermaid\s*(.*?)\s*```", block, flags=re.DOTALL)
        assert len(diagrams) == 1
        assert diagrams[0].splitlines()[0] == "flowchart TD"
        node_definitions = re.findall(
            r"\b([A-Za-z][A-Za-z0-9_]*)\[([^\]\n]+)\]", diagrams[0]
        )
        node_ids = [node_id for node_id, _ in node_definitions]
        assert node_ids
        assert len(node_ids) == len(set(node_ids))

    execution = _readme_section(readme, "## ⚙️ Lokale Ausführung")
    execution_blocks = _details_blocks(execution)
    assert len(execution_blocks) == 1
    assert (
        "<summary><strong>Lokale Ausführung und Deployment anzeigen</strong></summary>"
        in execution_blocks[0]
    )
    assert re.findall(r"^### (.+)$", execution_blocks[0], flags=re.MULTILINE) == [
        "Lokaler Start",
        "Streamlit Community Cloud",
        "Konfiguration",
        "OpenAI-Kontingent administrieren",
    ]


def test_readme_preserves_configuration_contract_and_resolvable_local_links():
    readme = Path("README.md").read_text(encoding="utf-8")
    execution = _readme_section(readme, "## ⚙️ Lokale Ausführung")
    documented_variables = set(
        re.findall(r"^\| `([A-Z][A-Z0-9_]*)` \|", execution, flags=re.MULTILINE)
    )
    template_variables = {
        name
        for line in Path(".env.template").read_text(encoding="utf-8").splitlines()
        if (name := line.partition("=")[0].strip())
        and re.fullmatch(r"[A-Z][A-Z0-9_]*", name)
    }
    assert documented_variables == template_variables

    runtime_configuration = Path(
        "src/configuration/configuration_runtime.py"
    ).read_text(encoding="utf-8")
    assert all(f'"{name}"' in runtime_configuration for name in template_variables)
    assert (
        "Der oben verlinkte Endpunkt ist die vorgesehene öffentliche Adresse"
        not in readme
    )

    markdown_targets = re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", readme)
    local_targets = []
    for target in markdown_targets:
        parsed = urlparse(target)
        if parsed.scheme or parsed.netloc or target.startswith("#"):
            continue
        local_targets.append(Path(unquote(parsed.path)))
    assert local_targets
    assert all(target.exists() for target in local_targets)


def test_readme_tree_matches_tracked_paths_and_compacts_tests():
    entries = _readme_project_tree_entries()
    documented_paths = [path for path, _, _ in entries]
    assert len(documented_paths) == len(set(documented_paths))

    tracked_paths = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    tracked_non_test_files = {
        path for path in tracked_paths if not path.startswith("tests/")
    }
    tracked_non_test_directories = {
        parent.as_posix()
        for path in tracked_non_test_files
        for parent in Path(path).parents
        if parent != Path(".")
    }

    documented_non_test_files = {
        path
        for path, is_directory, _ in entries
        if not is_directory and not path.startswith("tests/")
    }
    documented_non_test_directories = {
        path for path, is_directory, _ in entries if is_directory and path != "tests"
    }
    assert documented_non_test_files == tracked_non_test_files
    assert documented_non_test_directories == tracked_non_test_directories

    documented_tests = [entry for entry in entries if entry[0] == "tests"]
    assert documented_tests == [
        (
            "tests",
            True,
            "Deterministische Tests für Anwendung, Ingestion, Provider, "
            "Kontingente, Sessions und Vektorspeicher",
        )
    ]
    assert not any(path.startswith("tests/") for path in documented_paths)


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
