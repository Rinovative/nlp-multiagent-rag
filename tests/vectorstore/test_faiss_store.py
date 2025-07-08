import glob
import os
import pytest
from src.vectorstore.faiss_store import FAISSStore
from src.utils.utils import FileUtils

# Flag zum Löschen des Index nach dem Test
DELETE_AFTER_TEST = True

# Verzeichnis der Test-Dokumente
JSON_DIR = os.path.join(os.path.dirname(__file__), "..", "ingestion", "test_documents")

# Nur die .json-Dateien mit den zugehörigen Embeddings werden erfasst
JSON_FILES = glob.glob(
    os.path.join(JSON_DIR, "*.ingested.processed.chunked.embedded.json")
)

# Initialisiere den FAISSStore einmal, um den Index zu laden oder zu erstellen
faiss_store = FAISSStore(index_file="tests/vectorstore/test_faiss_index.index")


@pytest.fixture(
    scope="module"
)  # Die Fixture wird einmal für alle Tests im Modul ausgeführt
def setup_and_teardown():
    # Vor dem Testlauf
    print("📦 Test Setup abgeschlossen")

    yield  # Hier wird die Kontrolle an den Test übergeben

    # Nach dem Testlauf
    if DELETE_AFTER_TEST:
        faiss_store.delete_index()
        print("📂 FAISS-Index wurde nach dem Test gelöscht.")


@pytest.mark.parametrize("json_path", JSON_FILES)
def test_faiss_store_add_and_search(json_path, setup_and_teardown):
    print(f"📄 Test startet für: {json_path}")

    # Lade das JSON-Dokument mit Embeddings mit der FileUtils-Klasse
    doc = FileUtils.load_json(json_path)

    # Füge die Embeddings dem FAISS-Index hinzu
    faiss_store.process_json_and_add(doc)

    # Speichern des FAISS-Indexes explizit
    faiss_store.save_index()  # Speichert den FAISS-Index, falls er noch nicht gespeichert wurde

    # Prüfe, ob der Index gespeichert wurde
    assert os.path.exists(
        "tests/vectorstore/test_faiss_index.index"
    ), "FAISS-Index wurde nicht gespeichert."
    print("✔️ FAISS-Index wurde erfolgreich gespeichert.")

    # Suche nach einem ähnlichen Embedding (verwendet das erste Embedding als Beispiel)
    query_embedding = faiss_store.embeddings_list[
        0
    ]  # Verwende das erste Embedding aus dem gespeicherten Index

    # Verwende die `search` Methode anstatt `search_vektor` direkt
    results = faiss_store.search(query_embedding, k=3)

    # Ergebnisprüfungen
    assert isinstance(results, list), "Die Ergebnisse müssen eine Liste sein."
    assert len(results) == 3, "Es sollten genau 3 ähnliche Chunks gefunden werden."

    # Überprüfe, ob die Ergebnisse Chunk-IDs, Distanzen und Texte enthalten und gebe sie aus
    for result in results:
        assert "chunk_id" in result, "Ergebnisse müssen eine 'chunk_id' enthalten."
        assert "distance" in result, "Ergebnisse müssen eine 'distance' enthalten."
        assert "text" in result, "Ergebnisse müssen den 'text' enthalten."
        print(
            f"Chunk ID: {result['chunk_id']}, Distance: {result['distance']}, Text: {result['text']}"
        )

    print("✅ Test für FAISSStore erfolgreich abgeschlossen.")
