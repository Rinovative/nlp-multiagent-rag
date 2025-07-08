import faiss
import numpy as np
import os
from typing import List, Dict


class FAISSStore:
    def __init__(self, index_file: str = "faiss_index.index", dim: int = 1536):
        self.index_file = index_file
        self.index = None
        self.embeddings_list = []
        self.chunk_ids = []
        self.texts = []
        self.dim = dim

        if os.path.exists(self.index_file):
            try:
                self.load_index()
            except Exception:
                self.create_index()
        else:
            self.create_index()

    def create_index(self):
        dim = self.dim
        self.index = faiss.IndexFlatL2(dim)
        self.save_index()

    def load_index(self):
        self.index = faiss.read_index(self.index_file)

    def save_index(self):
        faiss.write_index(self.index, self.index_file)

    def delete_index(self):
        """
        Löscht den FAISS-Index und die zugehörige Index-Datei.
        """
        if os.path.exists(self.index_file):
            os.remove(self.index_file)

        # Rücksetzen des Indexes und der Daten
        self.index = None
        self.embeddings_list = []
        self.chunk_ids = []

    def extract_embeddings(self, doc: List[Dict]) -> List[Dict]:
        chunk_ids = []
        embeddings = []
        texts = []

        document_title = None
        file_hash = None

        for single_doc in doc:
            chunk_id = single_doc.get("chunk_id", "")

            if chunk_id == "metadata":
                document_title = single_doc.get("text", {}).get(
                    "document_title", "Unknown_Title"
                )
                file_hash = single_doc.get("text", {}).get("file_hash", "Unknown_Hash")
                continue

            if document_title is None or file_hash is None:
                continue

            embedding = single_doc.get("embedding", None)
            text = single_doc.get("text", "").strip()

            if embedding is not None:
                unique_chunk_id = f"{document_title}_{chunk_id}_{file_hash}"
                chunk_ids.append(unique_chunk_id)
                embeddings.append(embedding)
                texts.append(text)

        return chunk_ids, embeddings, texts

    def add_embeddings(
        self, embeddings: List[List[float]], chunk_ids: List[str], texts: List[str]
    ):
        if not embeddings:
            return

        embeddings_array = np.array(embeddings, dtype=np.float32)

        self.index.add(embeddings_array)

        self.embeddings_list.extend(embeddings)
        self.chunk_ids.extend(chunk_ids)
        self.texts.extend(texts)

        if len(self.embeddings_list) != len(self.chunk_ids):
            raise ValueError(
                "Die Anzahl der Embeddings stimmt nicht mit der Anzahl der Chunk-IDs überein!"
            )

        self.save_index()

    def process_json_and_add(self, doc: List[Dict]):
        chunk_ids, embeddings, texts = self.extract_embeddings(doc)

        if embeddings:
            self.add_embeddings(embeddings, chunk_ids, texts)

        self.save_index()

    def search_vektor(self, query_embedding: List[float], k: int = 3) -> List[Dict]:
        """
        Sucht die k nächsten Vektoren basierend auf der Benutzerabfrage.
        """
        query_array = np.array([query_embedding], dtype=np.float32)

        distances, indices = self.index.search(query_array, k)

        results = []
        for i in range(k):
            index = indices[0][i]
            if index < len(self.chunk_ids):
                chunk_id = self.chunk_ids[index]
                distance = distances[0][i]
                results.append({"chunk_id": chunk_id, "distance": distance})

        return results

    def get_text_by_chunk_id(self, chunk_id: str) -> str:
        """
        Gibt den Text basierend auf der chunk_id zurück.
        Nutzt ein Dictionary für schnelleren Zugriff.
        """
        chunk_text_dict = dict(zip(self.chunk_ids, self.texts))
        return chunk_text_dict.get(chunk_id, "Text nicht gefunden.")

    def search(self, query_embedding: List[float], k: int = 3) -> List[Dict]:
        """
        Sucht die k nächsten Vektoren basierend auf der Benutzerabfrage und gibt die Ergebnisse
        mit Chunk-ID, Distanz und zugehörigem Text zurück.
        """
        # Vektorsuche durchführen
        search_results = self.search_vektor(query_embedding, k)

        # Texte für gefundene chunk_ids abrufen
        results = []
        for result in search_results:
            chunk_id = result["chunk_id"]
            distance = result["distance"]

            # Text für die gefundene chunk_id holen
            text = self.get_text_by_chunk_id(chunk_id)

            # Ergebnis zusammenstellen
            results.append({"chunk_id": chunk_id, "distance": distance, "text": text})

        return results
