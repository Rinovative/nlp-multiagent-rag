"""
===============================================================================
ingestion_preprocessing.py
===============================================================================
Normalize and enrich extracted PDF structure for deterministic chunking.

Responsibilities:
  - Classify headings, captions, pseudo-tables, headers, and footers.
  - Enrich the stable document metadata consumed by chunking.

Design principles:
  - Apply deterministic classifiers in one documented mutation sequence.
  - Mutate only the caller-supplied loader mapping.

Boundaries:
  - Operates only on loader mappings and performs no I/O or embedding work.
  - Does not parse original PDF bytes or choose chunk sizes.
===============================================================================
"""

from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
import math
import re
from typing import Any

__all__ = ["PdfPreprocessor"]


class PdfPreprocessor:
    """Enrich the structured output produced by ``UniversalPDFLoader``.

    Parameters
    ----------
    json_data
        Caller-owned loader mapping to enrich in place.

    Notes
    -----
    Classifiers run in a fixed order and mutate the supplied mapping so chunking
    remains deterministic without copying a potentially large PDF representation.
    """

    def __init__(self, json_data: dict[str, Any]) -> None:
        """Create a preprocessor over one caller-owned loader mapping."""

        self.json_data = json_data
        self.paragraphs: list[dict[str, Any]] = self._collect_paragraphs()
        self.title: str | None = None
        self.font_stats: dict[float, int] = {}
        self.main_font_size: float | None = None
        self.header_footer_candidate_sizes: set[float] = set()

    def run_preprocessing(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run all structural classifiers in their required order.

        Returns
        -------
        tuple[dict[str, Any], dict[str, Any]]
            The enriched document and a summary of removed headers and footers.
        """

        self._analyze_font_sizes()
        self._extract_title()
        removed_info = self._detect_and_remove_headers_footers()
        self._detect_headings()
        self._detect_table_and_image_captions()
        self._detect_pseudo_tables()
        self.process_and_save_metadata()
        return self.json_data, removed_info

    def _collect_paragraphs(self):
        paragraphs = []
        for page in self.json_data.get("pages", []):
            paragraphs.extend(page.get("paragraphs", []))
        return paragraphs

    def _extract_title(self):
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
        font_counter: Counter[float] = Counter()
        font_character_counter: Counter[float] = Counter()
        for para in self.paragraphs:
            size = para.get("font_size", None)
            if size is not None:
                rounded_size = round(size, 2)
                font_counter[rounded_size] += 1
                font_character_counter[rounded_size] += max(
                    1, len(para.get("text", "").strip())
                )

        font_stats = dict(font_counter)
        main_font_size = None
        if font_character_counter:
            main_font_size = max(
                font_character_counter,
                key=lambda size: (
                    font_character_counter[size],
                    font_counter[size],
                    -size,
                ),
            )

        self.font_stats = font_stats
        self.main_font_size = main_font_size

        other_sizes = [
            (size, count)
            for size, count in font_stats.items()
            if size != main_font_size
        ]
        if main_font_size is None:
            top_nearby_sizes: list[float] = []
        else:
            other_sizes.sort(key=lambda item: abs(item[0] - main_font_size))
            top_nearby_sizes = [size for size, _ in other_sizes[:5]]
        self.header_footer_candidate_sizes = set(top_nearby_sizes)

        self.json_data.setdefault("metadata", {})
        self.json_data["metadata"]["font_size_stats"] = font_stats
        self.json_data["metadata"]["font_size_character_stats"] = dict(
            font_character_counter
        )
        self.json_data["metadata"]["main_font_size"] = main_font_size
        self.json_data["metadata"]["header_footer_candidate_sizes"] = sorted(
            self.header_footer_candidate_sizes
        )

    def _detect_headings(self, epsilon=0.1, min_length=4):
        if self.main_font_size is None:

            for page in self.json_data.get("pages", []):
                for para in page.get("paragraphs", []):
                    para["heading_level"] = 0
                    para["is_type"] = "normal"
            self.json_data.setdefault("metadata", {})
            self.json_data["metadata"]["headings"] = []
            return []

        heading_sizes = set()

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

        sorted_sizes = sorted(heading_sizes, reverse=True)

        size_to_level = {size: idx + 1 for idx, size in enumerate(sorted_sizes)}

        headings = []

        for page in self.json_data.get("pages", []):
            page_num = page.get("page", None)
            for para in page.get("paragraphs", []):
                size = round(para.get("font_size", 0), 2)
                text = para.get("text", "").strip()

                level = None

                if size > self.main_font_size + epsilon:
                    if len(text) >= min_length and not (
                        re.fullmatch(r"[\d]+[.)]?", text) or text in ["-", "•", "●"]
                    ):
                        level = size_to_level.get(size, None)
                        if level is None:
                            level = 1
                        para["heading_level"] = level
                        para["is_type"] = "heading"

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
                        para["is_type"] = "normal"

                elif abs(size - self.main_font_size) <= epsilon:
                    level = 0
                    para["is_type"] = "normal"

                elif size < self.main_font_size - epsilon:
                    level = -1
                    para["is_type"] = "normal"

                else:
                    level = 0
                    para["is_type"] = "normal"

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
        if self.main_font_size is None:
            return {}

        all_header_texts = []
        all_footer_texts = []
        header_text_map: dict[str, list[str]] = {}
        footer_text_map: dict[str, list[str]] = {}

        for page in self.json_data.get("pages", []):
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

            if combined_header:
                header_text_map.setdefault(combined_header, []).extend(header_texts)
            if combined_footer:
                footer_text_map.setdefault(combined_footer, []).extend(footer_texts)

        recognized_headers = self._cluster_repeated_texts(
            all_header_texts, similarity_threshold, occurrence_ratio
        )
        recognized_footers = self._cluster_repeated_texts(
            all_footer_texts, similarity_threshold, occurrence_ratio
        )

        removed_headers_candidates = []
        removed_footers_candidates = []
        removed_headers_fallback = []
        removed_footers_fallback = []

        header_paras_flat = set()
        for combined_text, para_list in header_text_map.items():
            if any(
                self._similarity(combined_text, recognized) >= similarity_threshold
                for recognized in recognized_headers
            ):
                header_paras_flat.update(para_list)

        footer_paras_flat = set()
        for combined_text, para_list in footer_text_map.items():
            if any(
                self._similarity(combined_text, recognized) >= similarity_threshold
                for recognized in recognized_footers
            ):
                footer_paras_flat.update(para_list)

        for page in self.json_data.get("pages", []):
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
        self, text_list, similarity_threshold, occurrence_ratio
    ):
        clusters: list[str] = []
        cluster_counts: list[int] = []

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

        min_occurrences = max(2, math.ceil(len(text_list) * occurrence_ratio))
        recognized = [
            clusters[i]
            for i, count in enumerate(cluster_counts)
            if count >= min_occurrences
        ]

        return recognized

    def _similarity(self, a, b):
        return SequenceMatcher(None, a, b).ratio()

    def get_metadata(self) -> dict[str, Any]:
        """Return the current document metadata mapping.

        Returns
        -------
        dict
            Current metadata, or a new empty mapping when metadata is absent.
        """

        return self.json_data.get("metadata", {})

    def _detect_table_and_image_captions(self, merge_threshold_factor=2):
        for page in self.json_data.get("pages", []):
            paras = page.get("paragraphs", [])
            last_para = None

            for para in paras:
                if para["heading_level"] > 0:
                    last_para = para
                    continue

                if last_para:
                    y_distance = abs(para["y_position"] - last_para["y_position"])

                    if (
                        y_distance > (last_para["font_size"] * merge_threshold_factor)
                        and para["heading_level"] < 0
                    ):
                        para["is_type"] = "caption"
                    else:
                        para["is_type"] = "normal"

                else:
                    para["is_type"] = "normal"

                last_para = para

        return self.json_data

    def _detect_pseudo_tables(self, look_ahead=3):
        pseudo_tables_per_page = []

        for page in self.json_data.get("pages", []):
            paragraphs = page.get("paragraphs", [])
            in_pseudo_table = False
            non_matching_count = 0

            for para in paragraphs:
                text = para.get("text", "").strip()

                if para.get("is_type") == "caption":
                    continue

                match = re.match(r".+\.\s*\.\s*\.\s*\.\s*.*", text)

                if match:

                    para["text"] = re.sub(r"(\.)(\s*\.){2,}", "\t", text)
                    para["is_type"] = "pseudo_table"
                    in_pseudo_table = True
                    non_matching_count = 0

                elif in_pseudo_table:
                    non_matching_count += 1
                    if non_matching_count > look_ahead:
                        break
                    para["is_type"] = "pseudo_table"

            if in_pseudo_table:

                unique_heading_levels = sorted(
                    {
                        para["heading_level"]
                        for para in paragraphs
                        if para.get("is_type") == "pseudo_table"
                    }
                )
                unique_pages = sorted(
                    {
                        page["page"]
                        for para in paragraphs
                        if para.get("is_type") == "pseudo_table"
                    }
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

        return pseudo_tables_per_page

    def process_and_save_metadata(self) -> dict[str, Any]:
        """Consolidate detected tables and captions into document metadata.

        Returns
        -------
        dict
            Updated metadata mapping attached to the caller-owned document.

        Notes
        -----
        Despite its historical name, this method mutates only the in-memory
        document mapping and does not write a file.
        """

        pseudo_tables = []
        tables = []
        captions = []

        if "tables" in self.json_data:

            tables = self.json_data["tables"]
            self.json_data["metadata"]["tables"] = tables
            del self.json_data["tables"]

        current_table = None

        for page in self.json_data["pages"]:
            for paragraph in page["paragraphs"]:

                if paragraph["is_type"] == "pseudo_table":
                    if current_table is None:

                        current_table = {
                            "text": paragraph["text"],
                            "pages": [page["page"]],
                            "heading_levels": [paragraph["heading_level"]],
                            "contents": [paragraph["text"]],
                        }
                    else:

                        current_table["text"] += " " + paragraph["text"]
                        current_table["pages"].append(page["page"])
                        current_table["heading_levels"].append(
                            paragraph["heading_level"]
                        )
                        current_table["contents"].append(paragraph["text"])
                else:
                    if current_table is not None:

                        current_table["heading_levels"] = sorted(
                            set(current_table["heading_levels"])
                        )
                        current_table["pages"] = sorted(set(current_table["pages"]))
                        pseudo_tables.append(current_table)
                        current_table = None

                if paragraph["is_type"] == "caption":
                    captions.append(
                        {
                            "text": paragraph["text"],
                            "page": page["page"],
                            "heading_levels": [paragraph["heading_level"]],
                        }
                    )

        if current_table:

            current_table["heading_levels"] = sorted(
                set(current_table["heading_levels"])
            )
            current_table["pages"] = sorted(set(current_table["pages"]))
            pseudo_tables.append(current_table)

        existing_metadata = self.json_data.get("metadata", {})

        existing_metadata["pseudo_tables"] = pseudo_tables
        existing_metadata["tables"] = tables
        existing_metadata["captions"] = captions

        self.json_data["metadata"] = existing_metadata

        return existing_metadata
