import glob
import os
import pytest
from pathlib import Path
from src.utils.utils import FileUtils  # Importiere die FileUtils-Klasse
from src.ingestion.embedder import PDFEmbedder
from dotenv import load_dotenv
import openai

# Verzeichnis der Test-Dokumente
PDF_DIR = "tests/ingestion/test_documents"

# Nur die .ingested.processed.chunked.json-Dateien werden erfasst
JSON_FILES = glob.glob(os.path.join(PDF_DIR, "*.ingested.processed.chunked.json"))

# Lade die Umgebungsvariablen
load_dotenv()

# API-Schlüssel aus der .env-Datei
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Überprüfe, ob der API-Schlüssel geladen wurde
if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY muss in der .env-Datei gesetzt sein.")

# OpenAI-API-Schlüssel setzen
openai.api_key = OPENAI_API_KEY


@pytest.mark.parametrize("json_path", JSON_FILES)
def test_pdf_embedder_processing(json_path):
    """
    Testet den PDFEmbedder, indem es ein JSON-Dokument verarbeitet, Chunks klassifiziert,
    die jeweiligen Metadaten verarbeitet und das Ergebnis als neues JSON speichert.
    """
    output_path = Path(json_path).with_suffix(".embedded.json")

    # Initialisiere den PDFEmbedder mit einem OpenAI API-Schlüssel
    embedder = PDFEmbedder(openai_key=OPENAI_API_KEY)

    # Lade das Dokument mit FileUtils
    doc = FileUtils.load_json(json_path)

    # Verarbeite das JSON-Dokument und erhalte Schritt-für-Schritt Ergebnisse
    result = []
    for processed_data, step in embedder.process_json(doc):
        # Du kannst hier `step` ausgeben, wenn du während der Verarbeitung Informationen sehen möchtest
        print(
            step
        )  # Diese Zeile kann entfernt oder durch andere Verarbeitungen ersetzt werden
        result.append(processed_data)

    # Speichern des Ergebnisses im Test
    FileUtils.save_json(result, output_path)

    # Prüfe, ob die Ausgabedatei gespeichert wurde
    assert os.path.exists(
        output_path
    ), f"Output-Datei wurde nicht gespeichert: {output_path}"

    print(f"✅ Neues JSON gespeichert unter: {output_path}")

    print("✅ Embedding erfolgreich abgeschlossen.")
