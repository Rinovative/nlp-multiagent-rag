from sentence_transformers import SentenceTransformer


class MemoryAgent:
    def __init__(self, memory_storage, max_history=5):
        """
        Initialisiert den MemoryAgent mit einem Speicher und einer maximalen Anzahl von zu speichernden Nachrichten.
        :param memory_storage: Das Objekt, das den Speicher verwaltet (Redis-basiert).
        :param max_history: Die maximale Anzahl von Nachrichten, die im Speicher behalten werden.
        """
        self.memory_storage = memory_storage
        self.max_history = max_history
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def add_to_memory(self, chat_id: str, role: str, content: str):
        """
        Speichert eine Nachricht im Speicher und begrenzt die Anzahl auf `max_history` Nachrichten.
        :param chat_id: Die ID des Gesprächs, um die Nachricht zu speichern.
        :param role: Die Rolle der Nachricht (z. B. "user", "assistant", "system").
        :param content: Der Inhalt der Nachricht.
        """
        try:
            # Hole den aktuellen Verlauf für das gegebene Chat-ID
            history = (
                self.memory_storage.get_memory(chat_id) or []
            )  # Lege einen leeren Verlauf an, falls nicht vorhanden

            # Füge die neue Nachricht zum Verlauf hinzu
            history.append({"role": role, "content": content})

            # Begrenze den Verlauf auf die letzten `max_history` Nachrichten
            if len(history) > self.max_history:
                history.pop(
                    0
                )  # Entferne die älteste Nachricht, wenn das Limit überschritten wird

            # Speichere den neuen Verlauf im Redis-Speicher
            self.memory_storage.set_memory(chat_id, history)

        except Exception as e:
            print(f"Fehler beim Speichern der Nachricht: {e}")

    def get_memory(self, chat_id: str):
        """
        Gibt alle Benutzeranfragen und die letzte Antwort des Assistenten zurück.

        :param chat_id: Die ID des Gesprächs, dessen Verlauf abgerufen werden soll
        :return: Eine Liste von relevanten Gedächtniseinträgen inklusive der Rolle
        """
        try:
            full_context = self.memory_storage.get_memory(chat_id) or []

            relevant_memory = []

            # Füge alle Benutzeranfragen hinzu
            for entry in full_context:
                if entry["role"] == "user":
                    relevant_memory.append(
                        {"role": entry["role"], "content": entry["content"]}
                    )

            # Finde die letzte Antwort des Assistenten
            last_assistant_response = None
            for entry in reversed(full_context):
                if entry["role"] == "assistant":
                    last_assistant_response = entry
                    break

            # Wenn eine Antwort des Assistenten existiert, füge sie hinzu
            if last_assistant_response:
                relevant_memory.append(last_assistant_response)

            return relevant_memory

        except Exception as e:
            print(f"Fehler beim Abrufen des Gedächtnisses: {e}")
            return []
