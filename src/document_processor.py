import time
from src.utils.utils import FileUtils
from src.ingestion.loader import UniversalPDFLoader
from src.ingestion.preprocessing import PdfPreprocessor
from src.ingestion.chunker import PDFChunker
from src.ingestion.embedder import PDFEmbedder


class DocumentProcessor:
    def __init__(self, openai_api_key, faiss_store, max_chunk_length=1000):
        """
        Initialisiert die Dokumentenverarbeitung mit den benötigten Parametern.

        Args:
            openai_api_key (str): Der OpenAI API-Schlüssel.
            faiss_store (FAISSStore): Eine Instanz der FAISSStore-Klasse, die den FAISS-Index verwaltet.
            max_chunk_length (int): Maximale Länge eines Chunks.
        """
        # Initialisierung der benötigten Komponenten
        self.loader = UniversalPDFLoader()
        self.chunker = PDFChunker(max_chunk_length=max_chunk_length)
        self.embedder = PDFEmbedder(openai_key=openai_api_key)
        self.faiss_store = faiss_store  # Übergebene FAISS-Instanz
        self.preprocessor = PdfPreprocessor

    def _create_temp_dir(self):
        """Stellt sicher, dass der temp-Ordner existiert."""
        FileUtils.create_temp_dir()

    def _save_pdf(self, file, file_path):
        """Speichert die hochgeladene PDF im temp-Verzeichnis."""
        FileUtils.save_pdf(file, file_path)

    def _process_pdf(self, uploaded_file):
        """Lädt das PDF, extrahiert Text und verarbeitet ihn."""
        self._create_temp_dir()

        # PDF speichern und laden
        temp_pdf_path = FileUtils.get_temp_pdf_path()
        self._save_pdf(uploaded_file, temp_pdf_path)

        # PDF laden und verarbeiten
        document_data = self.loader.load_pdf(temp_pdf_path, extract_tables=True)
        if not document_data:
            raise ValueError("Das Dokument konnte nicht verarbeitet werden.")

        return document_data

    def _preprocess_text(self, document_data):
        """Verarbeitet das Dokument und gibt die vorverarbeiteten Daten zurück."""
        processed_json, removed_info = self.preprocessor(
            document_data
        ).run_preprocessing()
        if not processed_json:
            raise ValueError("Die Vorverarbeitung des Dokuments ist fehlgeschlagen.")
        return processed_json

    def _chunk_text(self, processed_json):
        """Teilt den vorverarbeiteten Text in Chunks."""
        return self.chunker.chunk_document(processed_json)

    def _create_embeddings(self, chunks):
        """Erstellt Embeddings für die Chunks."""
        embedding = []
        for processed_data, step in self.embedder.process_json(chunks):
            embedding.append(processed_data)
        return embedding

    def _save_embeddings(self, embedding):
        """Speichert die Embeddings im temp-Verzeichnis mit einem Zeitstempel."""
        timestamp = int(time.time())
        temp_json_path = FileUtils.get_temp_json_path(timestamp)
        FileUtils.save_json(embedding, temp_json_path)

    def _update_faiss(self, embedding):
        """Aktualisiert den FAISS-Index mit den neuen Embeddings."""
        self.faiss_store.process_json_and_add(embedding)

    def process_document(self, uploaded_file):
        """Kombiniert alle Verarbeitungsschritte."""
        try:
            # Verarbeite das Dokument
            document_data = self._process_pdf(uploaded_file)

            # Vorverarbeite den Text
            processed_json = self._preprocess_text(document_data)

            # Chunks erstellen
            chunks = self._chunk_text(processed_json)

            # Embeddings erstellen
            embedding = self._create_embeddings(chunks)

            # Embeddings speichern
            self._save_embeddings(embedding)

            # FAISS-Index aktualisieren
            self._update_faiss(embedding)

            return document_data

        except Exception as e:
            print(f"Fehler beim Verarbeiten des Dokuments: {e}")
            return None
