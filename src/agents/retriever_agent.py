import openai
import numpy as np
from sentence_transformers import SentenceTransformer


class RetrieverAgent:
    def __init__(self, faiss_store, client):
        """
        Initialisiert den RetrieverAgent mit einer FAISS-Datenbank.

        :param faiss_store: FAISS-Datenbank zur Speicherung und Suche von Embeddings
        :param client: OpenAI-Client für die Analyse der Anfrage
        """
        self.faiss_store = faiss_store
        self.client = client
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def retrieve_documents(self, query, memory):
        """
        Verarbeitet die Benutzeranfrage und führt die semantische Suche durch.

        :param query: Die Benutzeranfrage
        :return: Der Kontext (Text) der relevanten Dokumente sowie der numerische Detailgrad
        """
        detail_level = self._should_be_detailed(
            query
        )  # Berechne den numerischen Detailgrad
        top_k = self._get_top_k(
            detail_level
        )  # Berechne `top_k` basierend auf dem Detailgrad (zwischen 0 und 5 Dokumenten)

        # Kombiniere die Benutzeranfrage mit dem Gedächtnis
        combined_query = f"{query}\n\nMemory:\n{memory}"

        # Erstelle das Embedding der kombinierten Abfrage
        query_embedding = self._create_embedding(combined_query)

        # Führe die semantische Suche in der Vektordatenbank durch
        search_results = self.faiss_store.search(query_embedding, k=top_k)

        # Extrahiere den relevanten Text aus den Suchergebnissen
        context = "\n".join(
            [result["text"] for result in search_results if "text" in result]
        )
        return context

    def _create_embedding(self, text):
        """
        Erstellt ein Embedding für den gegebenen Text mit der OpenAI API.
        """
        response = openai.embeddings.create(
            model="text-embedding-ada-002", input=text  # OpenAI Modell für Embeddings
        )
        embedding = response.data[0].embedding
        return np.array(embedding)

    def _should_be_detailed(self, query):
        """
        Entscheidet, wie detailliert die Antwort benötigt wird, basierend auf Sentence-BERT.

        :param query: Die Benutzeranfrage
        :return: Ein Wert, der den Detailgrad der Antwort angibt: 'high', 'medium', 'low', 'summary', 'technical'
        """
        # Detaillierte und andere Kategorien
        categories = ["high", "medium", "low", "summary", "technical", "conceptual"]

        # Vorverarbeitung der Anfrage
        query_embedding = self.model.encode(
            query
        )  # Generiere das Embedding der Anfrage
        category_embeddings = self.model.encode(
            categories
        )  # Generiere Embeddings für jede Kategorie

        # Berechne die Kosinusähnlichkeit
        similarities = np.dot(query_embedding, np.array(category_embeddings).T)
        max_similarity_index = np.argmax(
            similarities
        )  # Index der ähnlichsten Kategorie

        # Mapping der Ähnlichkeiten auf Kategorien
        return categories[max_similarity_index]

    def _get_top_k(self, detail_level):
        """
        Konvertiert den Detailgrad in eine Ganzzahl und berechnet `top_k` für die semantische Suche.

        :param detail_level: Der Detailgrad als String ('high', 'medium', 'low', etc.)
        :return: Der numerische Wert von `top_k`
        """
        # Mapping des Detailgrads zu numerischen Werten
        detail_level_map = {
            "high": 20,
            "medium": 12,
            "low": 4,
            "summary": 2,  # Keine Ergebnisse oder sehr wenige Ergebnisse
            "technical": 16,
            "conceptual": 8,
        }

        # Rückgabe des entsprechenden Werts, Standardwert ist 'medium'
        return detail_level_map.get(detail_level, 5)  # Default ist 'medium'
