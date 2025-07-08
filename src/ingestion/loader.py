"""
{
  "metadata": {
    "document_language": "unknown",  // Sprache des Dokuments (z.B. "de", "en")
    "document_title": null,  // Titel des Dokuments
    "author": "Rino Albertin",  // Autor des Dokuments
    "subject": null,  // Fachgebiet oder Thema (optional)
    "producer": "Microsoft® Word für Microsoft 365",  // Software, die das Dokument erstellt hat
    "created_at": "D:20250706030909+02'00'",  // Erstellungsdatum im Format D:JJJJMMTTHHMMSS+TZ
    "file_name": "struktur.pdf",  // Dateiname
    "file_path": "tests/ingestion/test_documents\\struktur.pdf",  // Pfad der Datei
    "file_size": 14611,  // Dateigröße in Bytes
    "file_hash": "7ade4dedf2d9a16a67e1f28be818914e",  // MD5-Hash der Datei
    "num_pages": 1,  // Anzahl der Seiten im Dokument
    "all_links": []  // Liste aller relevanten Links im Dokument (z.B. URLs)
  },
  "pages": [  // Liste der Seiten im Dokument
    {
      "page": 1,  // Seitenzahl
      "text": "X",  // Text auf der Seite
      "is_empty": false,  // Gibt an, ob die Seite leer ist
      "text_length": 1,  // Länge des Textes in Zeichen
      "page_hash": "0cc175b9c0f1b6a831c399e269772661",  // Hash der Seite (z.B. MD5)
      "paragraphs": [  // Liste der Absätze auf dieser Seite
        {
          "text": "X",  // Text des Absatzes
          "font_size": 11.040000000000077,  // Schriftgröße des Absatzes
          "font_names": [  // Schriftarten, die im Absatz verwendet werden
            "BCDFEE+Aptos",
            "BCDEEE+Aptos"
          ],
          "text_hash": "0cc175b9c0f1b6a831c399e269772661",  // Hash des Absatzes
          "y_position": 73.2,  // Position des Textes auf der Seite (y-Koordinate)
          "links": []  // Liste von Links im Absatz (z.B. URLs)
        }
      ],
      "links": []  // Allgemeine Links für die Seite
    }
  ],
  "tables": []  // Liste der Tabellen im Dokument
}
"""

import pdfplumber
import hashlib
import re
from langdetect import detect
import os
from urllib.parse import urlparse


class UniversalPDFLoader:
    def __init__(self):
        pass

    def _hash_text(self, text):
        """
        Beschreibung:
            Erstellt einen MD5-Hash eines Textes.

        Args:
            text (str): Eingabetext

        Returns:
            str: MD5-Hash-Wert
        """
        return hashlib.md5(text.strip().encode("utf-8")).hexdigest()

    def _safe_detect_language(self, sample_text, min_length=20):
        """
        Beschreibung:
            Erkennt die Sprache des Textes, sofern genügend Text vorhanden ist.

        Args:
            sample_text (str): Eingabetext zur Spracherkennung
            min_length (int): Minimale Länge des Texts

        Returns:
            str: ISO-Sprachcode (z.B. "de", "en") oder "unknown"
        """
        text = sample_text.strip()
        if not text or len(text) < min_length:
            return "unknown"
        return detect(text)

    def _compute_file_hash(self, path):
        """
        Beschreibung:
            Berechnet den MD5-Hash einer Datei.

        Args:
            path (str): Pfad zur Datei

        Returns:
            str: MD5-Hash der Datei
        """
        hasher = hashlib.md5()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _extract_pdf_metadata(self, pdfplumber_pdf):
        """
        Beschreibung:
            Extrahiert PDF-Metadaten aus dem Dokument.

        Args:
            pdfplumber_pdf (pdfplumber.PDF): Geöffnetes PDF-Objekt

        Returns:
            dict: Metadaten des Dokuments
        """
        meta = pdfplumber_pdf.metadata or {}
        return {
            "document_title": meta.get("Title", None),
            "author": meta.get("Author", None),
            "subject": meta.get("Subject", None),
            "producer": meta.get("Producer", None),
            "created_at": meta.get("CreationDate", None),
        }

    def _extract_annotation_links(self, page):
        """
        Beschreibung:
            Extrahiert klickbare Links aus PDF-Annotations-Objekten.

        Args:
            page (pdfplumber.page.Page): PDF-Seite

        Returns:
            set[str]: Menge der gefundenen Links
        """
        links = set()
        if hasattr(page, "annots") and page.annots:
            for annot in page.annots:
                uri = annot.get("uri")
                if uri:
                    links.add(uri)
        return links

    def _extract_text_links(self, text):
        """
        Beschreibung:
            Extrahiert URLs, die direkt im Text stehen.

        Args:
            text (str): Eingabetext

        Returns:
            list[str]: Liste gefundener Links
        """
        if not text:
            return []
        pattern = re.compile(r"(https?://[^\s\)\]}>\"]+)", re.IGNORECASE)
        return pattern.findall(text)

    def _extract_domain_from_url(self, url):
        """
        Beschreibung:
            Extrahiert die Domain aus einer URL (ohne www).

        Args:
            url (str): URL-String

        Returns:
            str | None: Domainname oder None, falls nicht extrahierbar
        """
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        domain = domain.lstrip("www.")
        return domain

    def _text_matches_domain(self, text, domain):
        """
        Beschreibung:
            Prüft, ob die Domain oder Teile davon im Text vorkommen.

        Args:
            text (str): Eingabetext
            domain (str): Domainname

        Returns:
            bool: True, falls Domain im Text erkannt wird
        """
        if not text or not domain:
            return False

        if domain in text.lower():
            return True

        domain_tokens = re.split(r"[.\-/]", domain)
        domain_tokens = [t for t in domain_tokens if t]
        for token in domain_tokens:
            if token and token.lower() in text.lower():
                return True

        return False

    def _extract_paragraphs_with_fonts(
        self, page, all_known_links, merge_threshold_factor=1.5, font_size_tolerance=0.2
    ):
        """
        Beschreibung:
            Liest eine PDF-Seite aus, gruppiert Textzeilen zu Paragraphen
            und weist Links anhand Domains zu.

        Args:
            page (pdfplumber.page.Page): PDF-Seite
            all_known_links (set[str]): Set aller Links auf dieser Seite
            merge_threshold_factor (float): Faktor für Zeilenabstand zur Paragraph-Trennung
            font_size_tolerance (float): Toleranz für Font-Size-Wechsel

        Returns:
            tuple[list[dict], set[str]]:
                Liste der Paragraphen,
                Menge aller Paragraphen zugeordneter Links
        """
        if not page.chars:
            return [], set()

        # Gruppiere alle Chars nach Y-Position
        y_positions = {}
        for char in page.chars:
            y_key = round(char["top"], 1)
            if y_key not in y_positions:
                y_positions[y_key] = []
            y_positions[y_key].append(char)

        sorted_ys = sorted(y_positions.keys())

        paragraphs = []
        current_para = None
        prev_y = None
        prev_font_size = None

        # Durchlaufe Zeilen von oben nach unten
        for y in sorted_ys:
            chars = y_positions[y]
            text_line = "".join([c["text"] for c in chars]).strip()
            if not text_line:
                continue

            max_size = max(c["size"] for c in chars)
            font_names = set(c["fontname"] for c in chars)

            new_para = False

            if current_para is None:
                new_para = True
            else:
                dy = abs(y - prev_y)
                threshold = prev_font_size * merge_threshold_factor
                font_size_diff = abs(max_size - prev_font_size)
                if dy > threshold or font_size_diff > font_size_tolerance:
                    new_para = True

            if new_para:
                if current_para:
                    paragraphs.append(current_para)
                current_para = {
                    "text": text_line,
                    "font_size": max_size,
                    "font_names": list(font_names),
                    "text_hash": self._hash_text(text_line),
                    "y_position": y,
                    "links": [],
                }
            else:
                current_para["text"] += " " + text_line

            prev_y = y
            prev_font_size = max_size

        if current_para:
            paragraphs.append(current_para)

        # (1) Direkt-Links im Text extrahieren
        assigned_links = set()
        for para in paragraphs:
            text_links = self._extract_text_links(para["text"])
            for link in text_links:
                if link not in para["links"]:
                    para["links"].append(link)
                    assigned_links.add(link)

        # (2) Links anhand Domain erkennen
        for uri in list(all_known_links):
            domain = self._extract_domain_from_url(uri)
            if not domain:
                continue

            for para in paragraphs:
                if self._text_matches_domain(para["text"], domain):
                    if uri not in para["links"]:
                        para["links"].append(uri)
                        assigned_links.add(uri)
                    break

        return paragraphs, assigned_links

    def _table_to_text(self, table):
        """
        Beschreibung:
            Wandelt eine Tabelle in Text um (Pipe-separiert).

        Args:
            table (list[list[str]]): Tabelleninhalt

        Returns:
            str: Tabelle als Text
        """
        rows = []
        for row in table:
            rows.append(" | ".join(cell if cell is not None else "" for cell in row))
        return "\n".join(rows)

    def load_pdf(self, pdf_path, extract_tables=True):
        """
        Beschreibung:
            Lädt ein PDF, extrahiert Text, Paragraphen, Tabellen
            und erkennt Links.

        Args:
            pdf_path (str): Pfad zur PDF-Datei
            extract_tables (bool): Ob Tabellen extrahiert werden sollen

        Returns:
            dict: Vollständige Analyse des Dokuments
        """
        pages_data = []
        tables_data = []

        file_size = os.stat(pdf_path).st_size
        file_hash = self._compute_file_hash(pdf_path)

        all_raw_texts = []
        all_links_global = set()
        global_assigned_links = set()

        with pdfplumber.open(pdf_path) as pdf:
            pdf_meta = self._extract_pdf_metadata(pdf)

            # Alle Links im Dokument sammeln
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                all_raw_texts.append(page_text)

                text_links = self._extract_text_links(page_text)
                all_links_global.update(text_links)

                annotation_links = self._extract_annotation_links(page)
                all_links_global.update(annotation_links)

            sample_text = "\n".join(all_raw_texts[:5])[:10000]
            document_language = self._safe_detect_language(sample_text)

            for i, page in enumerate(pdf.pages):
                page_text_links = set(
                    self._extract_text_links(page.extract_text() or "")
                )
                page_annotation_links = set(self._extract_annotation_links(page))
                page_links_on_this_page = page_text_links.union(page_annotation_links)

                # Prüfe nur Links, die noch nicht global in einem Paragraphen zugeordnet wurden
                remaining_links = page_links_on_this_page - global_assigned_links

                paragraphs, assigned_links = self._extract_paragraphs_with_fonts(
                    page, all_known_links=remaining_links
                )

                # Diese Links sind jetzt global in einem Paragraphen verbucht
                global_assigned_links.update(assigned_links)

                still_unassigned_links = remaining_links - assigned_links

                page_text = "\n".join([p["text"] for p in paragraphs])
                text_length = len(page_text)
                page_hash = hashlib.md5(page_text.encode("utf-8")).hexdigest()

                is_empty = len(page_text.strip()) == 0

                pages_data.append(
                    {
                        "page": i + 1,
                        "text": page_text,
                        "is_empty": is_empty,
                        "text_length": text_length,
                        "page_hash": page_hash,
                        "paragraphs": paragraphs,
                        "links": list(still_unassigned_links),
                    }
                )

                if extract_tables:
                    tables = page.extract_tables()
                    for table in tables:
                        cleaned_table = [
                            [cell if cell is not None else "" for cell in row]
                            for row in table
                        ]
                        if any(
                            any(cell.strip() for cell in row) for row in cleaned_table
                        ):
                            tables_data.append(
                                {
                                    "page": i + 1,
                                    "table": cleaned_table,
                                    "table_text": self._table_to_text(cleaned_table),
                                }
                            )

        result = {
            "metadata": {
                "document_language": document_language,
                "document_title": pdf_meta.get("document_title"),
                "author": pdf_meta.get("author"),
                "subject": pdf_meta.get("subject"),
                "producer": pdf_meta.get("producer"),
                "created_at": pdf_meta.get("created_at"),
                "file_name": os.path.basename(pdf_path),
                "file_path": pdf_path,
                "file_size": file_size,
                "file_hash": file_hash,
                "num_pages": len(pages_data),
                "all_links": sorted(all_links_global),
            },
            "pages": pages_data,
            "tables": tables_data,
        }

        return result
