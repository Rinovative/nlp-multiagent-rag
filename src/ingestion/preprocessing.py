"""
{
  "metadata": {
    "document_language": "unknown",  // Sprache des Dokuments (z.B. "de" für Deutsch, "en" für Englisch)
    "document_title": "X",  // Titel des Dokuments, hier ein Platzhalter
    "author": "Rino Albertin",  // Autor des Dokuments
    "subject": null,  // Fachgebiet oder Thema des Dokuments, falls vorhanden, sonst null
    "producer": "Microsoft® Word für Microsoft 365",  // Software, die das Dokument erstellt hat
    "created_at": "D:20250706030909+02'00'",  // Erstellungsdatum im speziellen Format (D:JJJJMMTTHHMMSS+TZ)
    "file_name": "struktur.pdf",  // Dateiname des Dokuments
    "file_path": "tests/ingestion/test_documents\\struktur.pdf",  // Pfad zur Datei
    "file_size": 14611,  // Dateigröße in Bytes
    "file_hash": "7ade4dedf2d9a16a67e1f28be818914e",  // MD5-Hash der Datei zur Identifikation
    "num_pages": 1,  // Anzahl der Seiten im Dokument
    "all_links": [],  // Liste aller Links im Dokument (z.B. URLs)
    "font_size_stats": {  // Statistik der Schriftgrößen im Dokument
      "11.04": 1  // Zeigt an, wie oft eine bestimmte Schriftgröße verwendet wurde (z.B. 11.04pt wurde 1-mal verwendet)
    },
    "main_font_size": 11.04,  // Die Hauptschriftgröße im Dokument
    "header_footer_candidate_sizes": [],  // Mögliche Größen von Headern und Fußzeilen, falls vorhanden
    "recognized_headers": [],  // Liste von erkannten Headern, z.B. falls bestimmte Titel als Header erkannt wurden
    "recognized_footers": [],  // Liste von erkannten Fußzeilen, falls vorhanden
    "headings": [  // Liste der Überschriften im Dokument
      {
        "text": "X",  // Text der Überschrift
        "level": 5,  // Überschriftsebene (z.B. 1 für Hauptüberschrift, 2 für Unterüberschrift, usw.)
        "font_size": 13.98,  // Schriftgröße der Überschrift
        "page": 1  // Seitenzahl, auf der die Überschrift vorkommt
      }
    ],
    "tables": [  // Liste der Tabellen im Dokument
      {
        "page": 1,  // Seitenzahl, auf der die Tabelle gefunden wurde
        "table": [  // Tabelle als 2D-Array
          [
            "X",  // Zelleninhalt der ersten Spalte
            "Y"   // Zelleninhalt der zweiten Spalte
          ]
        ],
        "table_text": "X | Y"  // Text der gesamten Tabelle (Zusammenfassung)
      }
    ],
    "pseudo_tables": [  // Liste für Pseudotabellen, die im Dokument als solche markiert wurden
      {
        "text": "X",  // Text der Pseudotabelle
        "pages": [ 1 ],  // Seitenzahl(en), auf der/dem die Pseudotabelle gefunden wurde
        "heading_levels": [ 0 ],  // Überschriftsebene(n), falls vorhanden
        "contents": ["X"]  // Inhalt der Pseudotabelle
      }
    ],
    "captions": [  // Liste für Bildunterschriften (z.B. für Tabellen oder Abbildungen)
      {
        "text": "X®",  // Text der Bildunterschrift
        "page": 1,  // Seitenzahl, auf der die Bildunterschrift zu finden ist
        "heading_levels": [ -1 ]  // Überschriftsebene(n) für die Bildunterschrift
      }
    ]
  },
  "pages": [  // Liste der Seiten im Dokument
    {
      "page": 211,  // Die Seitenzahl
      "text": "X",  // Text auf der Seite (hier als Platzhalter "X")
      "is_empty": false,  // Gibt an, ob die Seite leer ist (false bedeutet, die Seite enthält Text)
      "text_length": 593,  // Länge des Textes auf der Seite in Zeichen
      "page_hash": "cda3cd208dc5ff40976454b36682d9a6",  // Hash der Seite (z.B. MD5) zur Identifikation
      "paragraphs": [  // Liste der Absätze auf dieser Seite
        {
          "text": "X",  // Text des Absatzes (hier als Platzhalter "X")
          "font_size": 12.0,  // Schriftgröße des Absatzes
          "font_names": [  // Liste der verwendeten Schriftarten im Absatz
            "Arial-BoldItalicMT"
          ],
          "text_hash": "5fcfcb7df376059d0075cb892b2cc37f",  // Hash des Absatztextes
          "y_position": 77.1,  // Y-Position des Absatzes auf der Seite
          "links": [],  // Liste von Links im Absatz (falls vorhanden)
          "heading_level": 0,  // Überschriftsebene des Absatzes (0 für normalen Text)
          "is_type": "normal"  // Typ des Absatzes (normal, caption, pseudo_table)
        }
      ]
    }
  ]
}
"""

from collections import Counter
from difflib import SequenceMatcher
import re


class PdfPreprocessor:
    def __init__(self, json_data):
        """
        Initialisiert den PdfPreprocessor, der für die Vorverarbeitung des PDF-Daten-Dictionary verantwortlich ist.

        Args:
            json_data (dict): Eingabedaten im JSON-Format, die das Dokument repräsentieren.
        """
        self.json_data = json_data
        self.paragraphs = self._collect_paragraphs()
        self.title = None
        self.font_stats = {}
        self.main_font_size = None
        self.header_footer_candidate_sizes = set()

    def run_preprocessing(self):
        """
        Führt den gesamten Vorverarbeitungsprozess durch: Analysiert Schriftgrößen, extrahiert den Titel,
        erkennt Kopf- und Fußzeilen und speichert alle relevanten Metadaten.

        Returns:
            tuple: Das bearbeitete JSON-Daten-Dictionary und alle entfernten Header- und Footer-Informationen.
        """
        self._analyze_font_sizes()
        self._extract_title()
        removed_info = self._detect_and_remove_headers_footers()
        self._detect_headings()
        self._detect_table_and_image_captions()  # Bild- und Tabellenunterschriften erkennen
        self._detect_pseudo_tables()
        self.process_and_save_metadata()
        return self.json_data, removed_info

    def _collect_paragraphs(self):
        """
        Sammelt alle Absätze aus den Seiten des Dokuments.

        Returns:
            list: Eine Liste aller Absätze im Dokument.
        """
        paragraphs = []
        for page in self.json_data.get("pages", []):
            paragraphs.extend(page.get("paragraphs", []))
        return paragraphs

    def _extract_title(self):
        """
        Extrahiert den Titel des Dokuments, der als der größte Absatz mit der größten Schriftgröße
        betrachtet wird.

        Returns:
            str: Der Titel des Dokuments.
        """
        if not self.paragraphs:
            return None
        max_para = max(self.paragraphs, key=lambda para: para.get("font_size", 0))
        title_candidate = max_para.get("text", "").strip()
        if title_candidate:
            self.title = title_candidate
            self.json_data.setdefault("metadata", {})
            self.json_data["metadata"]["document_title"] = self.title
            return self.title
        else:
            return None

    def _analyze_font_sizes(self):
        """
        Analysiert die Schriftgrößen der Absätze und identifiziert die Hauptschriftgröße.
        Speichert alle Schriftgrößenstatistiken und ermittelt mögliche Header- und Footer-Schriftgrößen.

        Updates:
            - Speichert Schriftgrößenstatistiken und die Hauptschriftgröße in den Metadaten.
            - Ermittelt mögliche Header- und Footer-Schriftgrößen.
        """
        font_counter = Counter()
        for para in self.paragraphs:
            size = para.get("font_size", None)
            if size is not None:
                font_counter[round(size, 2)] += 1

        font_stats = dict(font_counter)
        main_font_size = None
        if font_stats:
            main_font_size = max(font_stats, key=font_stats.get)

        self.font_stats = font_stats
        self.main_font_size = main_font_size

        other_sizes = [
            (size, count)
            for size, count in font_stats.items()
            if size != main_font_size
        ]
        other_sizes.sort(key=lambda x: abs(x[0] - main_font_size))
        top_nearby_sizes = [size for size, _ in other_sizes[:5]]
        self.header_footer_candidate_sizes = set(top_nearby_sizes)

        self.json_data.setdefault("metadata", {})
        self.json_data["metadata"]["font_size_stats"] = font_stats
        self.json_data["metadata"]["main_font_size"] = main_font_size
        self.json_data["metadata"]["header_footer_candidate_sizes"] = sorted(
            self.header_footer_candidate_sizes
        )

    def _detect_headings(self, epsilon=0.1, min_length=4):
        """
        Erkennt Überschriften im Dokument und ordnet Paragraphen ein heading_level zu:
        - heading_level > 0 = echtes Heading
        - heading_level = 0 = Fließtext
        - heading_level < 0 = kleiner Text (z. B. Footnote)

        Speichert erkannte Headings zusätzlich in den Metadaten.

        Args:
            epsilon (float): Toleranz für Schriftgrößenabweichungen.
            min_length (int): Minimale Länge für gültige Überschriftstexte.

        Returns:
            list: Eine Liste von Überschriften, die Text, Level, Schriftgröße und Seitenzahl enthalten.
        """
        if self.main_font_size is None:
            # Keine Daten → alle Paragraphs = Fließtext
            for page in self.json_data.get("pages", []):
                for para in page.get("paragraphs", []):
                    para["heading_level"] = 0
                    para["is_type"] = "normal"  # Setze is_type auf normal für Fließtext
            self.json_data.setdefault("metadata", {})
            self.json_data["metadata"]["headings"] = []
            return []

        heading_sizes = set()

        # 1) Alle Größen größer als Fließtext sammeln (echte Headings)
        for para in self.paragraphs:
            size = para.get("font_size", 0)
            text = para.get("text", "").strip()

            if size <= self.main_font_size + epsilon:
                continue

            if len(text) < min_length:
                continue

            if re.fullmatch(r"[\d]+[.)]?", text) or text in ["-", "•", "●"]:
                continue

            heading_sizes.add(round(size, 2))

        # Sortieren, größte zuerst → Ebene 1
        sorted_sizes = sorted(heading_sizes, reverse=True)

        # Mapping Größe → Ebene
        size_to_level = {size: idx + 1 for idx, size in enumerate(sorted_sizes)}

        headings = []

        # 2) Nun alle Paragraphs durchgehen → Level setzen
        for page in self.json_data.get("pages", []):
            page_num = page.get("page", None)
            for para in page.get("paragraphs", []):
                size = round(para.get("font_size", 0), 2)
                text = para.get("text", "").strip()

                level = None

                # → Heading?
                if size > self.main_font_size + epsilon:
                    if len(text) >= min_length and not (
                        re.fullmatch(r"[\d]+[.)]?", text) or text in ["-", "•", "●"]
                    ):
                        level = size_to_level.get(size, None)
                        if level is None:
                            level = 1
                        para["heading_level"] = level
                        para["is_type"] = "heading"  # Setze is_type auf heading

                        heading = {
                            "text": text,
                            "level": level,
                            "font_size": size,
                            "page": page_num,
                        }
                        headings.append(heading)
                        continue
                    else:
                        level = 0
                        para["is_type"] = "normal"  # Setze is_type auf normal

                # → Fließtext
                elif abs(size - self.main_font_size) <= epsilon:
                    level = 0
                    para["is_type"] = "normal"  # Setze is_type auf normal

                # → kleiner als Fließtext
                elif size < self.main_font_size - epsilon:
                    level = -1
                    para["is_type"] = "normal"  # Setze is_type auf normal

                else:
                    level = 0
                    para["is_type"] = "normal"  # Setze is_type auf normal

                para["heading_level"] = level

        self.json_data.setdefault("metadata", {})
        self.json_data["metadata"]["headings"] = headings

        return headings

    def _detect_and_remove_headers_footers(
        self,
        header_threshold=100,
        footer_threshold=700,
        similarity_threshold=0.9,
        occurrence_ratio=0.5,
        merge_threshold_factor=2,
        short_text_len=10,
    ):
        """
        Erkennt und entfernt Kopf- und Fußzeilen anhand der Position und Schriftgröße der Absätze.

        Args:
            header_threshold (int): Schwellenwert für die Erkennung von Kopfzeilen.
            footer_threshold (int): Schwellenwert für die Erkennung von Fußzeilen.
            similarity_threshold (float): Ähnlichkeitsschwellenwert zur Erkennung wiederholter Texte.
            occurrence_ratio (float): Mindesthäufigkeit eines Textes, um als Header oder Footer erkannt zu werden.
            merge_threshold_factor (float): Faktor, um Absätze zu gruppieren.
            short_text_len (int): Minimale Länge eines Textes für die Erkennung als Header oder Footer.

        Returns:
            dict: Entfernte Header- und Footer-Informationen.
        """
        if self.main_font_size is None:
            return {}

        all_header_texts = []
        all_footer_texts = []
        header_text_map = {}
        footer_text_map = {}
        candidate_headers_per_page = []
        candidate_footers_per_page = []

        for page_idx, page in enumerate(self.json_data.get("pages", [])):
            paras = page.get("paragraphs", [])

            header_candidates, _ = self._detect_candidates(
                [p for p in paras if p["y_position"] < header_threshold],
                merge_threshold_factor,
            )
            footer_candidates, _ = self._detect_candidates(
                list(
                    reversed([p for p in paras if p["y_position"] > footer_threshold])
                ),
                merge_threshold_factor,
                reverse=True,
            )
            footer_candidates.reverse()

            header_texts = [p["text"].strip() for p in header_candidates]
            footer_texts = [p["text"].strip() for p in footer_candidates]

            combined_header = " ".join(header_texts).strip()
            combined_footer = " ".join(footer_texts).strip()

            all_header_texts.append(combined_header)
            all_footer_texts.append(combined_footer)

            candidate_headers_per_page.append(header_candidates)
            candidate_footers_per_page.append(footer_candidates)

            if combined_header:
                header_text_map.setdefault(combined_header, []).extend(header_texts)
            if combined_footer:
                footer_text_map.setdefault(combined_footer, []).extend(footer_texts)

        recognized_headers = self._cluster_repeated_texts(
            all_header_texts, similarity_threshold, occurrence_ratio, label="Header"
        )
        recognized_footers = self._cluster_repeated_texts(
            all_footer_texts, similarity_threshold, occurrence_ratio, label="Footer"
        )

        removed_headers_candidates = []
        removed_footers_candidates = []
        removed_headers_fallback = []
        removed_footers_fallback = []

        # Alle Einzeltexte flach sammeln
        header_paras_flat = set()
        for para_list in header_text_map.values():
            header_paras_flat.update(para_list)

        footer_paras_flat = set()
        for para_list in footer_text_map.values():
            footer_paras_flat.update(para_list)

        for page_idx, page in enumerate(self.json_data.get("pages", [])):
            new_paragraphs = []
            paras = page.get("paragraphs", [])

            for para in paras:
                text = para.get("text", "").strip()

                if text in header_paras_flat:
                    removed_headers_candidates.append(text)
                    continue
                if text in footer_paras_flat:
                    removed_footers_candidates.append(text)
                    continue

                new_paragraphs.append(para)

            page["paragraphs"] = new_paragraphs

        # Fallback nur, falls nichts gefunden wurde
        fallback_header_needed = len(recognized_headers) == 0
        fallback_footer_needed = len(recognized_footers) == 0

        if fallback_header_needed or fallback_footer_needed:
            for page in self.json_data.get("pages", []):
                paras = page["paragraphs"]
                remaining_paragraphs = []
                for para in paras:
                    text = para.get("text", "").strip()
                    size = para.get("font_size", 0)
                    y_pos = para.get("y_position", 0)

                    if size == self.main_font_size:
                        remaining_paragraphs.append(para)
                        continue

                    if (
                        fallback_header_needed
                        and y_pos < header_threshold
                        and len(text) <= short_text_len
                    ):
                        removed_headers_fallback.append(text)
                        continue

                    if (
                        fallback_footer_needed
                        and y_pos > footer_threshold
                        and len(text) <= short_text_len
                    ):
                        removed_footers_fallback.append(text)
                        continue

                    remaining_paragraphs.append(para)

                page["paragraphs"] = remaining_paragraphs

        self.json_data["metadata"]["recognized_headers"] = recognized_headers
        self.json_data["metadata"]["recognized_footers"] = recognized_footers

        return {
            "removed_headers_candidates": removed_headers_candidates,
            "removed_footers_candidates": removed_footers_candidates,
            "removed_headers_fallback": removed_headers_fallback,
            "removed_footers_fallback": removed_footers_fallback,
        }

    def _detect_candidates(self, paragraphs, merge_threshold_factor, reverse=False):
        """
        Hilfsfunktion zur Identifikation von Kandidaten für Kopf- und Fußzeilen anhand von Schriftgröße und Position.

        Args:
            paragraphs (list): Liste der Paragraphen, die überprüft werden.
            merge_threshold_factor (float): Faktor zur Bestimmung, wann Absätze zusammengeführt werden.
            reverse (bool): Flag, um die Richtung der Analyse umzukehren.

        Returns:
            tuple: Eine Liste von erkannten Kandidaten und die verbleibenden Paragraphen.
        """
        candidates = []
        remaining = []
        last_y = None

        for para in paragraphs:
            size = para.get("font_size", 0)
            y_pos = para.get("y_position", 0)

            if size == self.main_font_size:
                remaining.append(para)
                continue

            if size in self.header_footer_candidate_sizes:
                if last_y is None:
                    candidates.append(para)
                    last_y = y_pos
                    continue

                dy = abs(y_pos - last_y)
                threshold = size * merge_threshold_factor
                if dy > threshold:
                    break

                candidates.append(para)
                last_y = y_pos
            else:
                remaining.append(para)

        if reverse:
            candidates.reverse()

        return candidates, remaining

    def _cluster_repeated_texts(
        self, text_list, similarity_threshold, occurrence_ratio, label="Element"
    ):
        """
        Hilfsfunktion zur Erkennung wiederholter Texte im Dokument, die als Header oder Footer markiert werden.

        Args:
            text_list (list): Liste der zu überprüfenden Texte.
            similarity_threshold (float): Schwellenwert für die Ähnlichkeit von Texten.
            occurrence_ratio (float): Minimum an Häufigkeit, das ein Text erreichen muss, um als wiederholt zu gelten.
            label (str): Label für den Texttyp (z. B. "Header", "Footer").

        Returns:
            list: Eine Liste der erkannten wiederholten Texte.
        """
        clusters = []
        cluster_counts = []

        for text in text_list:
            if not text:
                continue

            matched = False
            for idx, cluster_text in enumerate(clusters):
                sim = self._similarity(text, cluster_text)
                if sim >= similarity_threshold:
                    cluster_counts[idx] += 1
                    matched = True
                    break

            if not matched:
                clusters.append(text)
                cluster_counts.append(1)

        min_occurrences = int(len(text_list) * occurrence_ratio)
        recognized = [
            clusters[i]
            for i, count in enumerate(cluster_counts)
            if count >= min_occurrences
        ]

        return recognized

    def _similarity(self, a, b):
        """
        Berechnet die Ähnlichkeit zwischen zwei Texten.

        Args:
            a (str): Der erste Text.
            b (str): Der zweite Text.

        Returns:
            float: Der Ähnlichkeitswert zwischen den beiden Texten.
        """
        return SequenceMatcher(None, a, b).ratio()

    def get_metadata(self):
        """
        Gibt die Metadaten des Dokuments zurück.

        Returns:
            dict: Das Metadaten-Dictionary des Dokuments.
        """
        return self.json_data.get("metadata", {})

    def _detect_table_and_image_captions(self, merge_threshold_factor=2):
        """
        Erkennung von Tabellen- und Bildunterschriften anhand des Abstands zwischen Absätzen und der Schriftgröße.
        Wenn der Abstand zum vorherigen Paragraphen groß ist und das Heading-Level kleiner ist, wird es als Caption klassifiziert.
        """
        for page in self.json_data.get("pages", []):
            paras = page.get("paragraphs", [])
            last_para = None

            for para in paras:
                # Wenn wir einen Paragraphen finden, der Abstand zum vorherigen hat und kleineres Heading-Level
                if last_para:
                    y_distance = abs(para["y_position"] - last_para["y_position"])

                    # Wenn der Abstand zu groß ist und das Heading-Level kleiner ist, wird es als Caption klassifiziert
                    if (
                        y_distance > (last_para["font_size"] * merge_threshold_factor)
                        and para["heading_level"] < 0
                    ):
                        para["is_type"] = "caption"  # Setze auf "caption"
                    else:
                        # Verhindere Überschreiben von is_type bei bereits gesetztem Heading
                        if (
                            para["heading_level"] > 0
                        ):  # Wenn es eine Überschrift ist, nicht auf "normal" setzen
                            continue
                        para["is_type"] = (
                            "normal"  # Setze auf "normal" für alle anderen
                        )

                else:
                    para["is_type"] = (
                        "normal"  # Falls der erste Paragraph, setze den Typ als 'normal'
                    )

                last_para = para  # Aktualisiere den letzten Paragraphen für den nächsten Vergleich

        return self.json_data

    def _detect_pseudo_tables(self, look_ahead=3):
        """
        Erkennt pseudo-tabellarische Strukturen wie Inhaltsverzeichnisse,
        Literaturverzeichnisse und Bildverzeichnisse.

        Sucht nach dem Muster 'xxx ... xxx' und prüft, ob es über mehrere Zeilen hinweg
        wiederholt auftritt und stoppt, wenn das Muster in den nächsten 3 Paragraphen nicht mehr auftaucht.
        """
        pseudo_tables_per_page = []  # Liste für die Pseudo-Tabellen pro Seite

        # Iteriere über jede Seite im Dokument
        for page in self.json_data.get("pages", []):
            paragraphs = page.get("paragraphs", [])  # Alle Paragraphen auf der Seite
            in_pseudo_table = False  # Flag, ob wir uns in einer Pseudo-Tabelle befinden
            non_matching_count = 0  # Zähler für Paragraphen ohne das Punktleiter-Muster

            # Durchlaufe alle Paragraphen auf der Seite
            for i, para in enumerate(paragraphs):
                text = para.get("text", "").strip()

                # Überprüfe, ob der Absatz bereits als Caption markiert wurde
                if para.get("is_type") == "caption":
                    continue  # Überspringe diesen Absatz, wenn es bereits eine Caption ist

                # Prüfe, ob der Text das Punktleiter-Muster enthält (z.B. "xxx ... xxx")
                match = re.match(r".+\.\s*\.\s*\.\s*\.\s*.*", text)

                if match:
                    # Ersetze die Punkte durch Tabulatoren
                    para["text"] = re.sub(r"(\.)(\s*\.){2,}", "\t", text)
                    para["is_type"] = "pseudo_table"  # Setze den Typ auf 'pseudo_table'
                    in_pseudo_table = True
                    non_matching_count = (
                        0  # Reset der Zählung der nicht passenden Paragraphen
                    )

                elif in_pseudo_table:
                    # Wenn der Paragraph zum Pseudo-Table-Muster gehört, behält er den Typ
                    if non_matching_count >= look_ahead:
                        break  # Stoppe die Sammlung, wenn das Muster nicht mehr gefunden wird
                    para["is_type"] = "pseudo_table"
                    non_matching_count = 0

                # Zähle Paragraphen, die nicht zum Muster gehören
                if "..." not in text and in_pseudo_table:
                    non_matching_count += 1

            if in_pseudo_table:
                # Speichern der Pseudo-Tabelle für die Seite, dabei nur eindeutige `heading_levels` und `pages`
                unique_heading_levels = list(
                    set(
                        para["heading_level"]
                        for para in paragraphs
                        if para.get("is_type") == "pseudo_table"
                    )
                )
                unique_pages = list(
                    set(
                        page["page"]
                        for para in paragraphs
                        if para.get("is_type") == "pseudo_table"
                    )
                )

                pseudo_tables_per_page.append(
                    {
                        "pseudo_table": [
                            para
                            for para in paragraphs
                            if para["is_type"] == "pseudo_table"
                        ],
                        "heading_levels": unique_heading_levels,
                        "pages": unique_pages,
                    }
                )

        return pseudo_tables_per_page  # Rückgabe der Pseudo-Tabellen

    def process_and_save_metadata(self):
        # Initiale Container für Pseudotabellen und Tabellen
        pseudo_tables = []
        tables = []
        captions = []

        # Tabellen in den Metadaten speichern
        if "tables" in self.json_data:
            # Übertrage Tabellen in die Metadaten
            tables = self.json_data["tables"]
            self.json_data["metadata"]["tables"] = tables
        del self.json_data["tables"]

        # Variablen für das Zusammenführen von Pseudotabellen
        current_table = None

        # Iteriere durch die Seiten und Absätze des Dokuments
        for page in self.json_data["pages"]:
            for paragraph in page["paragraphs"]:
                # Wenn der Absatz eine Pseudotabelle ist
                if paragraph["is_type"] == "pseudo_table":
                    if current_table is None:
                        # Beginne eine neue Pseudotabelle
                        current_table = {
                            "text": paragraph["text"],
                            "pages": [page["page"]],
                            "heading_levels": [paragraph["heading_level"]],
                            "contents": [paragraph["text"]],
                        }
                    else:
                        # Füge zur aktuellen Pseudotabelle hinzu
                        current_table["text"] += " " + paragraph["text"]
                        current_table["pages"].append(page["page"])
                        current_table["heading_levels"].append(
                            paragraph["heading_level"]
                        )
                        current_table["contents"].append(paragraph["text"])
                else:
                    if current_table is not None:
                        # Beende die aktuelle Pseudotabelle und füge sie hinzu
                        # Entferne Duplikate bei heading_levels und pages
                        current_table["heading_levels"] = list(
                            set(current_table["heading_levels"])
                        )
                        current_table["pages"] = list(set(current_table["pages"]))
                        pseudo_tables.append(current_table)
                        current_table = None

                # Alle Captions werden erfasst
                if paragraph["is_type"] == "caption":
                    captions.append(
                        {
                            "text": paragraph["text"],
                            "page": page["page"],
                            "heading_levels": [paragraph["heading_level"]],
                        }
                    )

        # Falls noch eine Pseudotabelle am Ende des Dokuments existiert
        if current_table:
            # Entferne Duplikate
            current_table["heading_levels"] = list(set(current_table["heading_levels"]))
            current_table["pages"] = list(set(current_table["pages"]))
            pseudo_tables.append(current_table)

        # Hole die bestehenden Metadaten, um sie zu beibehalten
        existing_metadata = self.json_data.get("metadata", {})

        # Füge die neuen Informationen zu den bestehenden Metadaten hinzu
        existing_metadata["pseudo_tables"] = pseudo_tables
        existing_metadata["tables"] = tables
        existing_metadata["captions"] = captions

        # Aktualisiere die Metadaten im JSON
        self.json_data["metadata"] = existing_metadata

        # Rückgabe der aktualisierten Metadaten
        return existing_metadata
