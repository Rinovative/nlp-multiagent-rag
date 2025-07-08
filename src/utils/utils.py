import os
import json
import numpy as np  # Nötig für np.float32, falls du es verwenden möchtest


class FileUtils:
    @staticmethod
    def create_temp_dir():
        """Stellt sicher, dass der temp-Ordner existiert."""
        if not os.path.exists("temp"):
            os.makedirs("temp")

    @staticmethod
    def save_pdf(file, file_path):
        """Speichert die hochgeladene PDF im temp-Verzeichnis."""
        try:
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        except Exception as e:
            print(f"Fehler beim Speichern der PDF: {e}")

    @staticmethod
    def save_json(data, file_path):
        """Speichert Daten als JSON und behandelt np.float32."""

        def convert_np_float32(obj):
            """Konvertiert np.float32 in float."""
            if isinstance(obj, np.float32):
                return float(obj)
            raise TypeError(
                f"Object of type {obj.__class__.__name__} is not JSON serializable"
            )

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(
                    data, f, default=convert_np_float32, indent=2, ensure_ascii=False
                )
        except Exception as e:
            print(f"Fehler beim Speichern der JSON-Datei: {e}")

    @staticmethod
    def load_json(input_path):
        """
        Lädt eine JSON-Datei und gibt die Daten zurück.

        Args:
            input_path (str): Pfad zur Eingabedatei

        Returns:
            dict: Die geladenen Daten aus der JSON-Datei
        """
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Fehler beim Laden der JSON-Datei: {e}")
            return None

    @staticmethod
    def get_temp_pdf_path(file_name="temp_document.pdf"):
        """Erzeugt den Pfad für das PDF im temp-Ordner."""
        return os.path.join("temp", file_name)

    @staticmethod
    def get_temp_json_path(timestamp):
        """Erzeugt den Pfad für die JSON-Datei im temp-Ordner mit Zeitstempel."""
        return os.path.join("temp", f"temp_path_{timestamp}.json")
