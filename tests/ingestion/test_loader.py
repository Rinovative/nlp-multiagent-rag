import glob
import os
import pytest
import logging
from pathlib import Path
from src.utils.utils import FileUtils

from src.ingestion.loader import UniversalPDFLoader

logging.basicConfig(level=logging.INFO)

PDF_DIR = "tests/ingestion/test_documents"


@pytest.mark.parametrize("pdf_path", glob.glob(os.path.join(PDF_DIR, "*.pdf")))
def test_ingest_generic(pdf_path):
    print(f"📄 Test startet für: {pdf_path}")

    loader = UniversalPDFLoader()
    result = loader.load_pdf(pdf_path, extract_tables=True)

    metadata = result["metadata"]
    pages = result["pages"]
    tables = result["tables"]

    # Metadata Checks
    assert isinstance(metadata, dict)
    assert "document_language" in metadata
    assert "file_name" in metadata
    assert "file_size" in metadata
    assert "file_hash" in metadata
    assert "num_pages" in metadata
    assert metadata["num_pages"] == len(pages)
    assert metadata["file_size"] > 0
    assert isinstance(metadata["file_hash"], str)

    # Page Checks
    assert isinstance(pages, list)
    assert len(pages) >= 1

    for p in pages:
        assert isinstance(p["page"], int)
        assert isinstance(p["text"], str)
        assert isinstance(p["is_empty"], bool)
        assert isinstance(p["text_length"], int)
        assert isinstance(p["page_hash"], str)
        assert isinstance(p["paragraphs"], list)

        for para in p["paragraphs"]:
            assert isinstance(para["text"], str)
            assert isinstance(para["font_size"], float) or isinstance(
                para["font_size"], int
            )
            assert isinstance(para["font_names"], list)

    # Table Checks (falls vorhanden)
    if tables:
        for t in tables:
            assert isinstance(t["page"], int)
            assert isinstance(t["table"], list)
            assert isinstance(t["table_text"], str)

            # Table darf nicht komplett leer sein
            assert any(any(cell.strip() for cell in row) for row in t["table"])

    # JSON speichern (Verwendung von FileUtils.save_json)
    dump_path = Path(pdf_path).with_suffix(".ingested.json")
    FileUtils.save_json(result, dump_path)  # Speichern der JSON mit FileUtils

    # JSON wieder laden zum Testen (Verwendung von FileUtils.load_json)
    reloaded = FileUtils.load_json(dump_path)  # Laden der JSON mit FileUtils
    assert "metadata" in reloaded
    assert "pages" in reloaded

    print(f"✅ JSON gespeichert unter {dump_path}")

    print("✅ Loading erfolgreich abgeschlossen.")
