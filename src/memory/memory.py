import redis
import json


class MemoryStorage:
    def __init__(self, redis_url, max_history: int = 5):
        """
        Initialisiert den Speicher mit einer Redis-Verbindung.

        :param redis_url: Die URL zur Redis-Datenbank.
        :param max_history: Die maximale Anzahl an gespeicherten Nachrichten pro Chat.
        """
        self.r = redis.from_url(redis_url)
        self.max_history = max_history

    def save_document(self, doc_id: str, document: dict):
        """
        Speichert ein gesamtes Dokument als JSON in Redis.

        :param doc_id: Die ID des Dokuments.
        :param document: Das Dokument, das gespeichert werden soll.
        """
        document_str = json.dumps(
            document
        )  # Umwandlung des Dokuments in einen JSON-String
        self.r.set(doc_id, document_str)  # Speichern des Dokuments als String in Redis

    def get_document(self, doc_id: str):
        """
        Holt ein gespeichertes Dokument aus Redis.

        :param doc_id: Die ID des Dokuments, das abgerufen werden soll.
        :return: Das Dokument als Python-Datenobjekt, wenn vorhanden, sonst None.
        """
        document_str = self.r.get(doc_id)  # Holt das gespeicherte Dokument als String
        if document_str:
            return json.loads(
                document_str
            )  # Umwandlung des JSON-Strings zurück in ein Python-Datenobjekt
        return None

    def get_memory(self, chat_id: str):
        """
        Holt den gespeicherten Verlauf für die angegebene Chat-ID aus Redis.

        :param chat_id: Die ID des Gesprächs, dessen Verlauf abgerufen werden soll.

        :return: Der Verlauf des Gesprächs als Liste oder ein leerer Verlauf, falls nicht vorhanden.
        """
        context = self.r.get(
            chat_id
        )  # Holt die gespeicherten Daten als String (falls vorhanden)
        if context:
            return json.loads(
                context
            )  # Wandelt den JSON-String in eine Python-Liste um
        return []

    def set_memory(self, chat_id: str, context: list):
        """
        Speichert den Verlauf für die angegebene Chat-ID in Redis und begrenzt die Anzahl an Nachrichten
        auf `max_history`.

        :param chat_id: Die ID des Gesprächs.
        :param context: Der Verlauf des Gesprächs, der gespeichert werden soll.
        """
        # Begrenze den Verlauf auf die letzten `max_history` Einträge
        context = context[
            -self.max_history :
        ]  # Nur die letzten `max_history` Nachrichten beibehalten

        self.r.set(
            chat_id, json.dumps(context)
        )  # Speichert die Liste als JSON-String in Redis
