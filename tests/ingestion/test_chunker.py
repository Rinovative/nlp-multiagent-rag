import glob
import os
import json
import pytest
import logging
from pathlib import Path
from src.utils.utils import FileUtils  # Importiere die FileUtils-Klasse
from src.ingestion.chunker import PDFChunker

logging.basicConfig(level=logging.INFO)

PDF_DIR = "tests/ingestion/test_documents"
JSON_FILES = glob.glob(os.path.join(PDF_DIR, "*.ingested.processed.json"))


@pytest.mark.parametrize("json_path", JSON_FILES)
def test_chunking(json_path):
    print(f"📄 Test startet für: {json_path}")

    # Lade das Dokument (hier als JSON, aber du könntest auch direkt ein PDF laden)
    with open(json_path, "r", encoding="utf-8") as file:
        doc = json.load(file)

    # Initialisiere den Chunker
    chunker = PDFChunker(max_chunk_length=1024)  # Maximaler Token-Limit für Chunks

    # Chunk das Dokument
    chunks = chunker.chunk_document(doc)

    # Test: Sicherstellen, dass Chunks erstellt wurden
    assert isinstance(chunks, list)
    assert len(chunks) > 0  # Sicherstellen, dass mindestens ein Chunk erstellt wurde

    print(f"✅ Chunking erfolgreich für {json_path} abgeschlossen.")

    # Test: Sicherstellen, dass alle chunk_ids eindeutig sind
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    duplicate_chunk_ids = [item for item in chunk_ids if chunk_ids.count(item) > 1]
    assert (
        not duplicate_chunk_ids
    ), f"Fehler: Es gibt doppelte chunk_ids: {set(duplicate_chunk_ids)}"

    # Speichern der Chunks mit der Utility-Klasse
    chunked_path = Path(json_path).with_suffix(".chunked.json")
    FileUtils.save_json(chunks, chunked_path)  # Jetzt über FileUtils speichern
    print(f"✅ Neues JSON gespeichert unter: {chunked_path}")

    print("✅ Chunking erfolgreich abgeschlossen.")
