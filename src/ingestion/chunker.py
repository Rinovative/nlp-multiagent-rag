"""[
  {
    "chunk_id": "metadata",
    "text": {
      "document_language": "de",  // Sprache des Dokuments
      "document_title": "Natural Language Processing (NLP) Projekt",  // Titel des Dokuments
      "author": "quant",  // Autor des Dokuments
      "subject": null,  // Thema des Dokuments
      "producer": "Microsoft® Word for Microsoft 365",  // Erstellende Software
      "created_at": "D:20250521140629+02'00'",  // Erstellungsdatum
      "file_name": "NLP Projekt Beschreibung 2025.pdf",  // Dateiname
      "file_path": "tests/ingestion/test_documents\\NLP Projekt Beschreibung 2025.pdf",  // Dateipfad
      "file_size": 302827,  // Dateigröße in Bytes
      "file_hash": "baecbdedfd8f41810d7fbc6041d8f8c6",  // Datei-Hash
      "num_pages": 6,  // Anzahl der Seiten im Dokument
      "all_links": [  // Alle Links im Dokument
        "https://..."
      ],
      "font_size_stats": {  // Schriftgrößenstatistik
        "17.04": 1,
        "12.0": 37,
        "8.04": 12,
      },
      "main_font_size": 12.0,  // Hauptschriftgröße
      "header_footer_candidate_sizes": [  // Mögliche Größen für Header und Footer
        8.04,
        9.96,
      ],
      "recognized_headers": [],  // Erkannte Header
      "recognized_footers": [  // Erkannte Footers
        "Projekt NLP– Frühlingssemester 2024/2025"
      ]
    }
  },
  {
    "chunk_id": "heading_1_page_1",  // Heading 1
    "text": "Natural Language Processing (NLP) Projekt",  // Text der Überschrift
    "page": 1,  // Seite
    "heading_level": 1 // Überschriftsebene
  },
  {
    "chunk_id": "table_page_1_structured",  // Table
    "text": [  // Inhalt der Tabelle als 2D-Array
      ["Text 1", "Text 2"],
      ["Text 1.1", "Text 2.1"]
    ],
    "page": 1  // Seite der Tabelle
  },
  {
    "chunk_id": "pseudo_table_page_1",  // Pseudo-Table
    "text": ["Text 1",
            "Text 2",] // Text der Pseudotabelle
    "pages": [1,2]  // Seite
    "heading_levels": [0, 1],  // Überschriftsebenen
  },
  {
    "chunk_id": "captions_0_page_2",  // Caption
    "text": "Text",
    "page": 2  // Seite
    "heading_level": -1  // Überschriftsebenen
  },
  {
    "chunk_id": "page_1_paragraph_1",  // Paragraph
    "text": "Natural Language Processing (NLP) Projekt",  // Text des Paragraphen
    "page": 1,  // Seite
    "heading_level": 0,  // Überschriftsebene
    "is_type": "normal", // Typ des Paragraphen (normal, heading, pseudo_table, caption)
  }
]
"""

import tiktoken
import json
from typing import List, Dict


class PDFChunker:
    def __init__(
        self,
        max_chunk_length: int = 1024,
        overlap_length: int = 256,
        min_chunk_length: int = 256,
    ):
        """
        Initialisiert den PDFChunker mit einer maximalen Chunklänge (in Tokens) und einer Überlappungslänge
        für den fließenden Übergang zwischen den Chunks.

        Args:
            max_chunk_length (int): Maximale Länge eines Chunks (in Tokens).
            overlap_length (int): Länge des Textes (in Tokens), der am Ende des vorherigen Chunks an den nächsten Chunk übergeben wird.
        """
        self.max_chunk_length = max_chunk_length
        self.overlap_length = overlap_length
        self.min_chunk_length = min_chunk_length
        self.encoder = tiktoken.get_encoding(
            "cl100k_base"
        )  # OpenAI Tokenizer für GPT-3/GPT-4
        self.document_title = None

    def _split_text_by_length(self, text: str) -> List[str]:
        """
        Splittet den Text in kleinere Teile, wenn er die maximale Länge überschreitet,
        wobei nur ein Teil des Endes des vorherigen Chunks an den nächsten Chunk angehängt wird,
        um fließende Übergänge zu gewährleisten. Wenn der letzte Chunk zu kurz ist, wird er an den vorherigen Chunk angehängt.

        Args:
            text (str): Der Text, der gesplittet werden soll.

        Returns:
            List[str]: Eine Liste von Text-Chunks, die die maximale Länge nicht überschreiten und fließende Übergänge haben.
        """
        tokenized_text = self.encoder.encode(text)
        chunks = []
        prev_chunk = ""  # Der Text des vorherigen Chunks

        while len(tokenized_text) > self.max_chunk_length:
            # Teile den Text in Chunks, die die maximale Länge nicht überschreiten
            chunk_part = tokenized_text[: self.max_chunk_length]
            chunk_text_part = self.encoder.decode(chunk_part)

            # Wenn der vorherige Chunk nicht leer ist, hänge einen Teil des Endes des vorherigen Chunks an
            if prev_chunk:
                chunk_text_part = prev_chunk[-self.overlap_length :] + chunk_text_part

            chunks.append(chunk_text_part)

            # Reduziere den tokenisierten Text um den aktuellen Chunk
            tokenized_text = tokenized_text[self.max_chunk_length :]

            # Der nächste Chunk startet mit dem Endteil des aktuellen
            prev_chunk = chunk_text_part[
                -self.max_chunk_length :
            ]  # Der letzte Teil für den nächsten Chunk

        # Verbleibender Text wird hinzugefügt
        if tokenized_text:
            chunk_text_part = self.encoder.decode(tokenized_text)
            chunks.append(chunk_text_part)

        # Wenn der letzte Chunk zu klein ist, füge ihn dem vorherigen hinzu
        if len(chunks[-1]) < self.min_chunk_length and len(chunks) > 1:
            chunks[-2] += chunks[-1]  # Füge den letzten Chunk dem vorherigen hinzu
            chunks = chunks[:-1]  # Entferne den jetzt leereren letzten Chunk

        return chunks

    def _split_table_by_length(
        self, structured_table: List[List[str]]
    ) -> List[List[str]]:
        """
        Splittet eine strukturierte Tabelle (als 2D-Array) in kleinere Teile, wenn sie die maximale Länge überschreitet,
        wobei der letzte Teil des vorherigen Chunks an den nächsten Chunk übergeben wird, um fließende Übergänge zu gewährleisten.

        Args:
            structured_table (List[List[str]]): Die strukturierte Tabelle (2D-Array), die in Chunks unterteilt werden soll.

        Returns:
            List[List[str]]: Eine Liste von Tabellen-Chunks, die die maximale Länge nicht überschreiten und fließende Übergänge haben.
        """
        # Die gesamte Tabelle als einen langen Text behandeln (als String), um die Länge zu überprüfen
        table_text = "\n".join(
            [" | ".join(row) for row in structured_table]
        )  # Verbinde jede Zeile mit '|' und dann alles mit '\n'

        # Berechne die Tokenlänge des gesamten Textes (nicht als JSON-String)
        total_length = len(self.encoder.encode(table_text))

        # Wenn die Gesamtlänge der Tabelle die maximale Länge überschreitet, teilen wir sie
        if total_length > self.max_chunk_length:
            chunks = []
            current_chunk = []
            current_length = 0
            prev_chunk_row = (
                None  # Um den Übergang von Zeilen zwischen Chunks zu ermöglichen
            )

            # Iteriere durch die Zeilen der Tabelle und füge sie in den aktuellen Chunk ein, solange es passt
            for row in structured_table:
                row_text = " | ".join(row)  # Verbinde die Zellen der Zeile
                row_length = len(
                    self.encoder.encode(row_text)
                )  # Berechne die Länge der Zeile in Tokens

                # Wenn das Hinzufügen der Zeile den aktuellen Chunk zu groß machen würde
                if current_length + row_length > self.max_chunk_length:
                    # Wenn der aktuelle Chunk nicht leer ist, füge ihn zu den Chunks hinzu
                    if current_chunk:
                        chunks.append(current_chunk)

                    # Starte einen neuen Chunk mit der aktuellen Zeile
                    current_chunk = [row]
                    current_length = row_length
                else:
                    # Wenn es noch passt, füge die Zeile zum aktuellen Chunk hinzu
                    current_chunk.append(row)
                    current_length += row_length

                # Wenn `prev_chunk_row` gesetzt ist, hänge einen Teil dieser Zeile an den nächsten Chunk an (Überlappung)
                if prev_chunk_row and current_length <= self.max_chunk_length:
                    # Achte darauf, dass nur 256 Tokens aus dem vorherigen Chunk übernommen werden
                    current_chunk[0] = prev_chunk_row + current_chunk[0]
                    prev_chunk_row = None  # Setze prev_chunk_row zurück

            # Füge den letzten Chunk hinzu, falls noch Zeilen übrig sind
            if current_chunk:
                chunks.append(current_chunk)

            # Überprüfe den letzten Chunk: Wenn er zu klein ist, füge ihn dem vorherigen hinzu
            if (
                len(chunks) > 1
                and len(
                    self.encoder.encode(
                        " | ".join([" | ".join(row) for row in chunks[-1]])
                    )
                )
                < self.min_chunk_length
            ):
                chunks[-2] += chunks[-1]  # Füge den letzten Chunk dem vorherigen hinzu
                chunks = chunks[:-1]  # Entferne den letzten Chunk

            return chunks
        else:
            # Wenn die gesamte Tabelle klein genug ist, gebe sie als 2D-Array zurück
            return [structured_table]

    def _chunk_metadata(self, metadata: Dict) -> List[Dict]:
        """
        Teilt die Metadaten in einen einzigen Chunk auf, wenn sie zu lang sind, aber behält die Struktur bei.

        Args:
            metadata (Dict): Die Metadaten des Dokuments.

        Returns:
            List[Dict]: Eine Liste mit einem oder mehreren Chunks der Metadaten.
        """
        # Kopiere die Metadaten, um sie nicht zu verändern
        metadata_copy = metadata.copy()

        # Entfernen von "captions", "tables", "pseudo_tables" und "headings", falls diese nicht benötigt werden
        for field in ["captions", "tables", "pseudo_tables", "headings"]:
            metadata_copy.pop(field, None)  # Entfernt Felder, wenn sie vorhanden sind

        # Die Metadaten in einen JSON-String umwandeln, um die Größe zu überprüfen
        metadata_json = json.dumps(metadata_copy)

        # Wenn die Länge des JSON-Strings zu groß ist, teilen wir ihn in kleinere Chunks
        if len(self.encoder.encode(metadata_json)) > self.max_chunk_length:
            # Wenn der gesamte JSON-Text zu lang ist, teilen wir ihn
            chunked_metadata = self._split_text_by_length(metadata_json)
            chunks = []
            for i, chunk in enumerate(chunked_metadata):
                chunks.append(
                    {
                        "chunk_id": f"{self.document_title}_metadata_chunk_{i}",
                        "text": chunk,
                    }
                )
        else:
            # Wenn der gesamte JSON-Text klein genug ist, behalten wir ihn in einem einzigen Chunk
            chunks = [
                {
                    "chunk_id": f"{self.document_title}_metadata",
                    "text": metadata_copy,
                }
            ]

        return chunks

    def _chunk_headings(self, headings: List[Dict]) -> List[Dict]:
        """
        Teilt Überschriften in Chunks auf, wenn sie zu lang sind.

        Args:
            headings (List[Dict]): Die Liste der Überschriften.

        Returns:
            List[Dict]: Eine Liste von Chunks der Überschriften.
        """
        chunks = []
        for idx, heading in enumerate(headings):  # Index pro heading hinzufügen
            heading_text = heading["text"]
            split_headings = self._split_text_by_length(heading_text)
            for split_heading in split_headings:
                chunk = {
                    "chunk_id": f"{self.document_title}_heading_{idx}_level_{heading['level']}_page_{heading['page']}",  # Index und level in chunk_id
                    "text": split_heading,
                    "page": heading["page"],
                    "heading_level": heading["level"],
                }
                chunks.append(chunk)
        return chunks

    def _chunk_tables(self, tables: List[Dict]) -> List[Dict]:
        """
        Teilt Tabellen in Chunks auf, wenn sie zu lang sind.

        Args:
            tables (List[Dict]): Die Liste der Tabellen.

        Returns:
            List[Dict]: Eine Liste von Chunks der Tabellen.
        """
        chunks = []
        for idx, table in enumerate(tables):  # Index pro Tabelle hinzufügen
            # Zugriff auf den 'table'-Schlüssel und dessen Inhalt (die Tabelle als 2D-Array)
            structured_table = table["table"]

            # Umwandlung der Tabelle in JSON und Prüfung der Länge
            structured_table_json = json.dumps(structured_table)
            if len(self.encoder.encode(structured_table_json)) > self.max_chunk_length:
                # Wenn die Tabelle zu lang ist, teilen wir sie in kleinere Teile
                split_texts = self._split_table_by_length(structured_table)
                for split_idx, split_text in enumerate(split_texts, 1):
                    chunk = {
                        "chunk_id": f"{self.document_title}_table_{idx}_page_{table['page']}_structured_{split_idx}",
                        "text": split_text,
                        "page": table["page"],
                    }
                    chunks.append(chunk)
            else:
                # Wenn die Tabelle klein genug ist, speichern wir sie als JSON
                chunk = {
                    "chunk_id": f"{self.document_title}_table_{idx}_page_{table['page']}_structured",  # Index hinzufügen
                    "text": json.loads(structured_table_json),
                    "page": table["page"],
                }
                chunks.append(chunk)

        return chunks

    def _chunk_pseudo_tables(self, pseudo_tables: List[Dict]) -> List[Dict]:
        """
        Teilt Pseudotabellen (die eine andere Struktur haben) in Chunks auf, wenn sie zu lang sind.

        Args:
            pseudo_tables (List[Dict]): Die Liste der Pseudotabellen.

        Returns:
            List[Dict]: Eine Liste von Chunks der Pseudotabellen.
        """
        chunks = []
        for idx, pseudo_table in enumerate(
            pseudo_tables
        ):  # Index pro Pseudotabelle hinzufügen
            # Zugriff auf den Inhalt der Pseudotabelle
            structured_pseudo_table = pseudo_table[
                "contents"
            ]  # Diese sind die Inhalte der Pseudotabelle

            # Wenn der Inhalt zu lang ist, teilen wir ihn auf
            structured_pseudo_table_json = json.dumps(structured_pseudo_table)
            if (
                len(self.encoder.encode(structured_pseudo_table_json))
                > self.max_chunk_length
            ):
                split_texts = self._split_table_by_length(structured_pseudo_table)
                for split_idx, split_text in enumerate(split_texts, 1):
                    chunk = {
                        "chunk_id": f"{self.document_title}_pseudo_table_{idx}_page_{pseudo_table['pages'][0]}_structured_{split_idx}",
                        "text": split_text,
                        "pages": pseudo_table["pages"],
                        "heading_levels": pseudo_table["heading_levels"],
                    }
                    chunks.append(chunk)
            else:
                # Wenn der Inhalt der Pseudotabelle klein genug ist, speichern wir sie als JSON
                chunk = {
                    "chunk_id": f"{self.document_title}_pseudo_table_{idx}_page_{pseudo_table['pages'][0]}_structured",  # Index hinzufügen
                    "text": json.loads(structured_pseudo_table_json),
                    "pages": pseudo_table["pages"],
                    "heading_levels": pseudo_table["heading_levels"],
                }
                chunks.append(chunk)

        return chunks

    def _chunk_captions(self, captions: List[Dict]) -> List[Dict]:
        """
        Teilt captions in Chunks auf, wenn sie zu lang sind.

        Args:
            captions (List[Dict]): Die Liste der captions.

        Returns:
            List[Dict]: Eine Liste von Chunks der captions.
        """
        chunks = []
        for idx, caption in enumerate(captions):  # Index pro caption hinzufügen
            heading_text = caption["text"]
            split_headings = self._split_text_by_length(heading_text)
            for split_heading in split_headings:
                chunk = {
                    "chunk_id": f"{self.document_title}_captions_{idx}_page_{caption['page']}",  # Index wird jetzt Teil der chunk_id
                    "text": split_heading,
                    "page": caption["page"],
                    "heading_level": caption["heading_levels"],
                }
                chunks.append(chunk)
        return chunks

    def chunk_document(self, doc: Dict) -> List[Dict]:
        """
        Teilt das Dokument in kleinere Chunks basierend auf Absätzen, Überschriften, Tabellen etc.

        Args:
            doc (Dict): Das Dokument, das geparsed wurde (mit Seiten und Absätzen).

        Returns:
            List[Dict]: Eine Liste von Chunks, die aus den Absätzen und Überschriften bestehen.
        """
        chunks = []

        # Metadaten des Dokuments
        metadata = doc.get("metadata", {})
        document_title = metadata.get("document_title", "Unknown_Title")
        self.document_title = document_title.replace(" ", "_")
        chunks.extend(self._chunk_metadata(metadata))

        # Überschriften
        headings = doc.get("metadata", {}).get("headings", [])
        chunks.extend(self._chunk_headings(headings))

        # Tabellen
        tables = doc.get("metadata", {}).get("tables", [])
        chunks.extend(self._chunk_tables(tables))

        # Pseudotabellen
        pseudo_tables = doc.get("metadata", {}).get("pseudo_tables", [])
        chunks.extend(self._chunk_pseudo_tables(pseudo_tables))

        captions = doc.get("metadata", {}).get("captions", [])
        chunks.extend(self._chunk_captions(captions))

        # Absätze auf den Seiten
        for page in doc.get("pages", []):
            page_num = page["page"]
            for paragraph in page["paragraphs"]:
                chunk_text = paragraph["text"]
                token_count = len(self.encoder.encode(chunk_text))
                if token_count > self.max_chunk_length:
                    split_texts = self._split_text_by_length(chunk_text)
                    for split_text in split_texts:
                        chunk = {
                            "chunk_id": f"{self.document_title}_page_{page_num}_paragraph_{page['paragraphs'].index(paragraph) + 1}",
                            "text": split_text,
                            "page": page_num,
                            "heading_level": paragraph["heading_level"],
                            "is_type": paragraph["is_type"],
                        }
                        chunks.append(chunk)
                else:
                    chunk = {
                        "chunk_id": f"{self.document_title}_page_{page_num}_paragraph_{page['paragraphs'].index(paragraph) + 1}",
                        "text": chunk_text,
                        "page": page_num,
                        "heading_level": paragraph["heading_level"],
                        "is_type": paragraph["is_type"],
                    }
                    chunks.append(chunk)

        return chunks
