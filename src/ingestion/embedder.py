from typing import List, Dict
from openai import OpenAI


class PDFEmbedder:
    def __init__(self, model: str = "text-embedding-ada-002", openai_key: str = None):
        self.openai_key = openai_key
        self.model = model

        if not self.openai_key:
            raise ValueError("OpenAI API key must be provided.")

        # Initialisiere den OpenAI-Client mit dem API-Key
        self.client = OpenAI(api_key=self.openai_key)

    def create_embeddings(self, text: str) -> List[float]:
        """Sende den Text an OpenAI und erhalte die Embeddings."""
        response = self.client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding

    def classify_chunk(self, chunk):
        """
        Klassifiziert den Chunk anhand seiner Struktur.

        Args:
            chunk (dict): Der Chunk, der klassifiziert werden soll.

        Returns:
            str: Der Typ des Chunks ("metadata", "heading", "table", "pseudo_table", "caption", "paragraph")
        """
        chunk_id = chunk.get("chunk_id", "")

        # Überprüfe, ob es sich um Metadaten handelt
        if chunk_id == "metadata":
            return "metadata"

        # Überprüfe, ob es sich um eine Überschrift handelt
        elif "heading" in chunk_id:
            return "heading"

        # Überprüfe, ob es sich um eine Pseudotabelle handelt
        elif "pseudo_table" in chunk_id:
            return "pseudo_table"

        # Überprüfe, ob es sich um eine Tabelle handelt
        elif "table" in chunk_id:
            return "table"

        # Überprüfe, ob es sich um eine Bildunterschrift handelt
        elif "captions" in chunk_id:
            return "caption"

        # Wenn `is_type` vorhanden ist und "normal" ist, handelt es sich um einen Paragraphen
        is_type = chunk.get("is_type", "")
        if is_type == "normal":
            return "paragraph"

        return "unknown"  # Wenn der Typ nicht identifiziert werden kann

    def _process_heading(
        self, document_title: str, text: str, page: int, heading_level: int
    ) -> str:
        """
        Verarbeitet eine Überschrift und strukturiert den Text zusammen mit den Metadaten.

        Args:
            document_title (str): Der Titel des Dokuments (z.B. "HAPSITE ER").
            text (str): Der Text der Überschrift.
            page (int): Die Seite, auf der die Überschrift erscheint.
            heading_level (int): Die Überschriftsebene (z.B. 1 für Hauptüberschrift).

        Returns:
            str: Der strukturierte Text der Überschrift zusammen mit den Metadaten.
        """
        structured_text = f"{document_title} - Heading (Page {page}, Heading Level {heading_level},): {text}"
        return structured_text

    def _process_table(
        self, document_title: str, text: List[List[str]], page: int
    ) -> str:
        """
        Verarbeitet eine Tabelle und strukturiert den Text zusammen mit den Metadaten.

        Args:
            document_title (str): Der Titel des Dokuments (z.B. "HAPSITE ER").
            chunk_id (str): Die ID des Chunks (z.B. "table_page_1_structured").
            text (List[List[str]]): Der Inhalt der Tabelle als 2D-Array (Liste von Listen).
            page (int): Die Seite, auf der die Tabelle erscheint.

        Returns:
            str: Der strukturierte Text der Tabelle zusammen mit den Metadaten.
        """
        structured_text = f"{document_title} - Table (Page {page}): "
        for row in text:
            structured_text += "Row: " + " | ".join(row) + "; "
        return structured_text.strip()

    def _process_pseudo_table(
        self,
        document_title: str,
        text: List[str],
        pages: List[int],
        heading_levels: List[int],
    ) -> str:
        """
        Verarbeitet den Text von Pseudotabellen (einfache Textliste) und strukturiert ihn zusammen mit den Metadaten.

        Args:
            document_title (str): Der Titel des Dokuments (z.B. "HAPSITE ER").
            chunk_id (str): Die ID des Chunks (z.B. "pseudo_table_page_1").
            text (List[str]): Der Text der Pseudotabelle (Liste von Strings).
            pages (List[int]): Die Seitenzahlen, auf denen die Pseudotabelle erscheint.
            heading_levels (List[int]): Die Überschriftsebenen des Textes.

        Returns:
            str: Der strukturierte Text der Pseudotabelle.
        """
        structured_text = f"{document_title} - Pseudotable (Pages {', '.join(map(str, pages))}, Heading Levels {', '.join(map(str, heading_levels))}): "
        for i, line in enumerate(text):
            structured_text += f"Line {i + 1}: {line}; "
        return structured_text.strip()

    def _process_caption(
        self, document_title: str, text: str, page: int, heading_level: int
    ) -> str:
        """
        Verarbeitet den Text von Bildunterschriften (Captions) und strukturiert ihn zusammen mit den Metadaten.

        Args:
            document_title (str): Der Titel des Dokuments (z.B. "HAPSITE ER").
            chunk_id (str): Die ID des Chunks (z.B. "captions_0_page_2").
            text (str): Der Text der Bildunterschrift.
            page (int): Die Seite, auf der die Bildunterschrift erscheint.
            heading_level (int): Die Überschriftsebene der Bildunterschrift.

        Returns:
            str: Der strukturierte Text der Bildunterschrift.
        """
        structured_text = f"{document_title} - Caption (Page {page}, Heading Level {heading_level}): {text}"
        return structured_text

    def _process_paragraph(
        self, document_title: str, text: str, page: int, heading_level: int
    ) -> str:
        """
        Verarbeitet den Text eines Paragraphen und strukturiert ihn zusammen mit den Metadaten.

        Args:
            document_title (str): Der Titel des Dokuments (z.B. "HAPSITE ER").
            chunk_id (str): Die ID des Chunks (z.B. "page_1_paragraph_1").
            text (str): Der Text des Paragraphen.
            page (int): Die Seite, auf der der Paragraph erscheint.
            heading_level (int): Die Überschriftsebene des Paragraphen.

        Returns:
            str: Der strukturierte Text des Paragraphen.
        """
        structured_text = f"{document_title} - Paragraph (Page {page}, Heading Level {heading_level}): {text}"
        return structured_text

    def process_json(self, doc: List[Dict]):
        """
        Liest das JSON und extrahiert die Textinhalte, generiert Embeddings.

        Args:
            doc (List[Dict]): Liste von Chunks, die verarbeitet werden sollen.

        Yields:
            dict: Verarbeitetes Ergebnis (einschließlich Embedding).
            str: Schritt der Verarbeitung
        """
        results = []
        document_title = "Unknown Document"  # Standardwert für den Fall, dass kein Titel extrahiert werden kann

        # Durch jedes Element in der Liste von Chunks iterieren
        for single_doc in doc:
            chunk_id = single_doc.get("chunk_id", "")

            # Wenn es sich um 'metadata' handelt, behandeln wir es speziell
            if "_metadata" in chunk_id:
                metadata = single_doc.get("text", {})

                # Extrahiere den Dokumenttitel aus den Metadaten
                document_title = metadata.get("document_title", document_title)

                # Entfernen von Feldern wie "captions", "tables", "pseudo_tables", "headings"
                for field in ["captions", "tables", "pseudo_tables", "headings"]:
                    metadata.pop(
                        field, None
                    )  # Entfernt Felder, wenn sie vorhanden sind

                # Füge Metadaten als ersten Eintrag hinzu
                results.append({"chunk_id": "metadata", "text": metadata})
                yield results[
                    -1
                ], "Processed metadata"  # Verarbeiteten Schritt sofort zurückgeben

            else:
                # Ansonsten jedes Chunk behandeln (z.B. headings, tables, etc.)
                text = single_doc.get("text", "")
                page = single_doc.get("page", 0)
                heading_level = single_doc.get("heading_level", 0)

                # Überprüfen und verarbeiten des Typs des Chunks
                chunk_type = self.classify_chunk(single_doc)

                # Verarbeite Text und Embedding
                if chunk_type == "heading":
                    processed_text = self._process_heading(
                        document_title, text, page, heading_level
                    )
                    embedding = self.create_embeddings(processed_text)
                    results.append(
                        {
                            "chunk_id": chunk_id,
                            "text": processed_text,
                            "embedding": embedding,
                        }
                    )
                    yield results[
                        -1
                    ], f"Processed heading for chunk_id: {chunk_id} | First part of text: {processed_text[:100]} | First part of embedding: {embedding[:3]}"

                elif chunk_type == "caption":
                    processed_text = self._process_caption(
                        document_title, text, page, heading_level
                    )
                    embedding = self.create_embeddings(processed_text)
                    results.append(
                        {
                            "chunk_id": chunk_id,
                            "text": processed_text,
                            "embedding": embedding,
                        }
                    )
                    yield results[
                        -1
                    ], f"Processed caption for chunk_id: {chunk_id} | First part of text: {processed_text[:100]} | First part of embedding: {embedding[:3]}"

                elif chunk_type == "pseudo_table":
                    pages = single_doc.get("pages", [])
                    heading_levels = single_doc.get("heading_levels", [])
                    processed_text = self._process_pseudo_table(
                        document_title, text, pages, heading_levels
                    )
                    embedding = self.create_embeddings(processed_text)
                    results.append(
                        {
                            "chunk_id": chunk_id,
                            "text": processed_text,
                            "embedding": embedding,
                        }
                    )
                    yield results[
                        -1
                    ], f"Processed pseudo_table for chunk_id: {chunk_id} | First part of text: {processed_text[:100]} | First part of embedding: {embedding[:3]}"

                elif chunk_type == "table":
                    processed_text = self._process_table(document_title, text, page)
                    embedding = self.create_embeddings(processed_text)
                    results.append(
                        {
                            "chunk_id": chunk_id,
                            "text": processed_text,
                            "embedding": embedding,
                        }
                    )
                    yield results[
                        -1
                    ], f"Processed table for chunk_id: {chunk_id} | First part of text: {processed_text[:100]} | First part of embedding: {embedding[:3]}"

                elif chunk_type == "paragraph":
                    processed_text = self._process_paragraph(
                        document_title, text, page, heading_level
                    )
                    embedding = self.create_embeddings(processed_text)
                    results.append(
                        {
                            "chunk_id": chunk_id,
                            "text": processed_text,
                            "embedding": embedding,
                        }
                    )
                    yield results[
                        -1
                    ], f"Processed paragraph for chunk_id: {chunk_id} | First part of text: {processed_text[:100]} | First part of embedding: {embedding[:3]}"

        return results
