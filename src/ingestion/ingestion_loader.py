"""
===============================================================================
ingestion_loader.py
===============================================================================
Extract deterministic structured content from PDF inputs.

Responsibilities:
  - Accept in-memory, file-like, or path-based PDF sources.
  - Extract metadata, paragraphs, links, and optional tables.
  - Produce the stable preprocessing input schema and SHA-256 identity.

Design principles:
  - Derive identities from bytes and avoid shared temporary upload files.
  - Preserve seekable caller-owned stream positions when possible.

Boundaries:
  - Does not classify document structure, chunk text, or create embeddings.
  - Does not perform OCR or persist uploaded bytes.
===============================================================================
"""

from __future__ import annotations

import hashlib
import io
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pdfplumber
from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

__all__ = ["UniversalPDFLoader"]


class UniversalPDFLoader:
    """Extract deterministic structured records from PDF inputs.

    Notes
    -----
    Uploaded bytes remain in memory. Seekable caller-owned streams are restored to
    their original position when possible, and document IDs are SHA-256 digests of
    the exact input bytes.
    """

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    @staticmethod
    def _safe_detect_language(sample_text: str, min_length: int = 20) -> str:
        text = sample_text.strip()
        if not text or len(text) < min_length:
            return "unknown"
        try:
            DetectorFactory.seed = 0
            return detect(text)
        except LangDetectException:
            return "unknown"

    @staticmethod
    def _compute_file_hash(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _prepare_pdf_source(
        self, pdf_source: Any, file_name: str | None = None
    ) -> tuple[Any, str, str | None, int, str]:
        if isinstance(pdf_source, (bytes, bytearray, memoryview)):
            data = bytes(pdf_source)
            if not data:
                raise ValueError("PDF content must not be empty.")
            return (
                io.BytesIO(data),
                file_name or "uploaded.pdf",
                None,
                len(data),
                hashlib.sha256(data).hexdigest(),
            )

        if hasattr(pdf_source, "read"):
            original_position = (
                pdf_source.tell() if hasattr(pdf_source, "tell") else None
            )
            data = pdf_source.read()
            if original_position is not None and hasattr(pdf_source, "seek"):
                pdf_source.seek(original_position)
            if not isinstance(data, (bytes, bytearray)) or not data:
                raise ValueError("PDF stream must provide non-empty bytes.")
            resolved_name = file_name or getattr(pdf_source, "name", "uploaded.pdf")
            return (
                io.BytesIO(bytes(data)),
                os.path.basename(str(resolved_name)),
                None,
                len(data),
                hashlib.sha256(bytes(data)).hexdigest(),
            )

        path = Path(pdf_source)
        if not path.is_file():
            raise FileNotFoundError(f"PDF file does not exist: {path}")
        return (
            str(path),
            file_name or path.name,
            str(path),
            path.stat().st_size,
            self._compute_file_hash(path),
        )

    @staticmethod
    def _extract_pdf_metadata(pdf: Any) -> dict[str, Any]:
        metadata = pdf.metadata or {}
        return {
            "document_title": metadata.get("Title"),
            "author": metadata.get("Author"),
            "subject": metadata.get("Subject"),
            "producer": metadata.get("Producer"),
            "created_at": metadata.get("CreationDate"),
        }

    @staticmethod
    def _extract_annotation_links(page: Any) -> set[str]:
        links: set[str] = set()
        for annotation in getattr(page, "annots", None) or []:
            uri = annotation.get("uri")
            if isinstance(uri, str) and uri:
                links.add(uri)
        return links

    @staticmethod
    def _extract_text_links(text: str) -> list[str]:
        if not text:
            return []
        pattern = re.compile(r'(https?://[^\s\)\]}>"]+)', re.IGNORECASE)
        return pattern.findall(text)

    @staticmethod
    def _extract_domain_from_url(url: str) -> str | None:
        domain = urlparse(url).netloc.casefold()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain or None

    @staticmethod
    def _text_matches_domain(text: str, domain: str) -> bool:
        if not text or not domain:
            return False
        normalized = text.casefold()
        if domain in normalized:
            return True
        return any(
            token.casefold() in normalized
            for token in re.split(r"[.\-/]", domain)
            if token
        )

    def _extract_paragraphs_with_fonts(
        self,
        page: Any,
        all_known_links: set[str],
        merge_threshold_factor: float = 1.5,
        font_size_tolerance: float = 0.2,
    ) -> tuple[list[dict[str, Any]], set[str]]:
        if not page.chars:
            return [], set()

        lines_by_y: dict[float, list[dict[str, Any]]] = {}
        for character in page.chars:
            lines_by_y.setdefault(round(character["top"], 1), []).append(character)

        paragraphs: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        previous_y: float | None = None
        previous_font_size: float | None = None
        for y_position in sorted(lines_by_y):
            characters = lines_by_y[y_position]
            text_line = "".join(character["text"] for character in characters).strip()
            if not text_line:
                continue
            font_size = max(character["size"] for character in characters)
            font_names = {character["fontname"] for character in characters}
            starts_paragraph = current is None
            if previous_y is not None and previous_font_size is not None:
                vertical_gap = abs(y_position - previous_y)
                font_difference = abs(font_size - previous_font_size)
                starts_paragraph = (
                    vertical_gap > previous_font_size * merge_threshold_factor
                    or font_difference > font_size_tolerance
                )

            if starts_paragraph:
                if current is not None:
                    current["text_hash"] = self._hash_text(current["text"])
                    paragraphs.append(current)
                current = {
                    "text": text_line,
                    "font_size": font_size,
                    "font_names": sorted(font_names),
                    "text_hash": self._hash_text(text_line),
                    "y_position": y_position,
                    "links": [],
                }
            else:
                assert current is not None
                current["text"] += f" {text_line}"
                current["font_names"] = sorted(set(current["font_names"]) | font_names)
            previous_y = y_position
            previous_font_size = font_size

        if current is not None:
            current["text_hash"] = self._hash_text(current["text"])
            paragraphs.append(current)

        assigned_links: set[str] = set()
        for paragraph in paragraphs:
            for link in self._extract_text_links(paragraph["text"]):
                if link not in paragraph["links"]:
                    paragraph["links"].append(link)
                    assigned_links.add(link)

        for uri in sorted(all_known_links):
            domain = self._extract_domain_from_url(uri)
            if domain is None:
                continue
            for paragraph in paragraphs:
                if self._text_matches_domain(paragraph["text"], domain):
                    if uri not in paragraph["links"]:
                        paragraph["links"].append(uri)
                        assigned_links.add(uri)
                    break
        return paragraphs, assigned_links

    @staticmethod
    def _table_to_text(table: list[list[str]]) -> str:
        return "\n".join(
            " | ".join(cell if cell is not None else "" for cell in row)
            for row in table
        )

    def load_pdf(
        self,
        pdf_source: Any,
        extract_tables: bool = True,
        file_name: str | None = None,
    ) -> dict[str, Any]:
        """Load a PDF into the stable preprocessing input schema.

        Parameters
        ----------
        pdf_source
            PDF bytes, a readable binary stream, or a filesystem path.
        extract_tables
            Whether to retain non-empty tables from each page.
        file_name
            Original upload name for in-memory sources.

        Returns
        -------
        dict
            Stable document mapping containing identity, metadata, pages, text,
            links, paragraphs, and optional tables.

        Raises
        ------
        TypeError
            If the source is not bytes, a readable binary stream, or a path.
        ValueError
            If byte or stream content is empty or unreadable.
        FileNotFoundError
            If an explicit filesystem path does not identify a file.

        Notes
        -----
        PDF parser exceptions propagate to ``DocumentProcessor``, which translates
        them into the project-owned UI-safe processing error.
        """

        (
            pdf_input,
            resolved_file_name,
            resolved_file_path,
            file_size,
            file_hash,
        ) = self._prepare_pdf_source(pdf_source, file_name=file_name)
        pages_data: list[dict[str, Any]] = []
        tables_data: list[dict[str, Any]] = []
        all_links: set[str] = set()
        globally_assigned_links: set[str] = set()

        with pdfplumber.open(pdf_input) as pdf:
            pdf_metadata = self._extract_pdf_metadata(pdf)
            raw_page_texts: list[str] = []
            page_links: list[set[str]] = []
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                raw_page_texts.append(page_text)
                links = set(self._extract_text_links(page_text))
                links.update(self._extract_annotation_links(page))
                page_links.append(links)
                all_links.update(links)

            language_sample = "\n".join(raw_page_texts[:5])[:10_000]
            document_language = self._safe_detect_language(language_sample)
            for page_index, page in enumerate(pdf.pages):
                remaining_links = page_links[page_index] - globally_assigned_links
                paragraphs, assigned_links = self._extract_paragraphs_with_fonts(
                    page, all_known_links=remaining_links
                )
                globally_assigned_links.update(assigned_links)
                unassigned_links = remaining_links - assigned_links
                page_text = "\n".join(paragraph["text"] for paragraph in paragraphs)
                pages_data.append(
                    {
                        "page": page_index + 1,
                        "text": page_text,
                        "is_empty": not page_text.strip(),
                        "text_length": len(page_text),
                        "page_hash": hashlib.sha256(
                            page_text.encode("utf-8")
                        ).hexdigest(),
                        "paragraphs": paragraphs,
                        "links": sorted(unassigned_links),
                    }
                )
                if extract_tables:
                    for table in page.extract_tables():
                        cleaned = [
                            [cell if cell is not None else "" for cell in row]
                            for row in table
                        ]
                        if any(any(cell.strip() for cell in row) for row in cleaned):
                            tables_data.append(
                                {
                                    "page": page_index + 1,
                                    "table": cleaned,
                                    "table_text": self._table_to_text(cleaned),
                                }
                            )

        return {
            "metadata": {
                "document_language": document_language,
                **pdf_metadata,
                "file_name": os.path.basename(str(resolved_file_name)),
                "file_path": resolved_file_path,
                "file_size": file_size,
                "file_hash": file_hash,
                "num_pages": len(pages_data),
                "all_links": sorted(all_links),
            },
            "pages": pages_data,
            "tables": tables_data,
        }
