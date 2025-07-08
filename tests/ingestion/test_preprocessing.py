import glob
import json
import os
import pytest
import logging
from pathlib import Path
from collections import Counter
from src.utils.utils import FileUtils  # Importiere die FileUtils-Klasse

from src.ingestion.preprocessing import PdfPreprocessor

logging.basicConfig(level=logging.INFO)

PDF_DIR = "tests/ingestion/test_documents"
JSON_FILES = glob.glob(os.path.join(PDF_DIR, "*.ingested.json"))


def summarize_removed(items, label):
    """
    Fasst entfernte Texte zusammen und gibt nur eindeutige Texte + Anzahl aus.
    """
    counter = Counter(items)
    if counter:
        print(f"⚠️ Entfernte {label}:")
        for text, count in counter.items():
            snippet = text if len(text) < 80 else text[:80] + "..."
            print(f"   - [{count}x] {snippet}")
    else:
        print(f"✅ Keine entfernten {label}.")


@pytest.mark.parametrize("json_path", JSON_FILES)
def test_preprocessing_generic(json_path):
    print(f"\n🔎 Test startet für JSON: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    pre = PdfPreprocessor(json_data)
    processed_json, removed_info = pre.run_preprocessing()

    meta = processed_json.get("metadata", {})

    assert "document_title" in meta
    title = meta.get("document_title")
    assert title is None or isinstance(title, str)
    if title:
        print(f"✅ Erkannter Titel: {title}")
    else:
        print("⚠️ Kein Titel erkannt (evtl. kleines PDF)")

    assert "font_size_stats" in meta
    font_stats = meta.get("font_size_stats", {})
    assert isinstance(font_stats, dict)
    print(f"✅ Schriftgrößen-Statistik: {font_stats}")

    assert "main_font_size" in meta
    main_font_size = meta.get("main_font_size")
    assert (
        main_font_size is None
        or isinstance(main_font_size, float)
        or isinstance(main_font_size, int)
    )
    print(f"✅ Haupt-Fließtext-Schriftgröße: {main_font_size}")

    candidate_sizes = meta.get("header_footer_candidate_sizes", [])
    print(f"✅ Kandidaten-Schriftgrößen für Header/Footer: {candidate_sizes}")

    assert "recognized_headers" in meta
    recognized_headers = meta.get("recognized_headers", [])
    assert isinstance(recognized_headers, list)
    print(f"✅ Erkannte Header: {recognized_headers}")

    assert "recognized_footers" in meta
    recognized_footers = meta.get("recognized_footers", [])
    assert isinstance(recognized_footers, list)
    print(f"✅ Erkannte Footer: {recognized_footers}")

    # Entfernte Inhalte schön ausgeben
    removed_headers_candidates = removed_info.get("removed_headers_candidates", [])
    removed_footers_candidates = removed_info.get("removed_footers_candidates", [])
    removed_headers_fallback = removed_info.get("removed_headers_fallback", [])
    removed_footers_fallback = removed_info.get("removed_footers_fallback", [])

    summarize_removed(removed_headers_candidates, "Header (Candidates)")
    summarize_removed(removed_footers_candidates, "Footer (Candidates)")
    summarize_removed(removed_headers_fallback, "Header (Fallback)")
    summarize_removed(removed_footers_fallback, "Footer (Fallback)")

    pages = processed_json.get("pages", [])
    for page in pages:
        paragraphs = page.get("paragraphs", [])
        assert isinstance(paragraphs, list)
        assert all(isinstance(p, dict) for p in paragraphs)

    for page in pages:
        for para in page.get("paragraphs", []):
            text = para.get("text", "").strip()
            for header in recognized_headers:
                if header:
                    assert (
                        text != header
                    ), f"⚠️ Header-Paragraph noch vorhanden: {header}"
            for footer in recognized_footers:
                if footer:
                    assert (
                        text != footer
                    ), f"⚠️ Footer-Paragraph noch vorhanden: {footer}"

    # Neues JSON speichern mit FileUtils
    processed_path = Path(json_path).with_suffix(".processed.json")
    FileUtils.save_json(
        processed_json, processed_path
    )  # Jetzt über FileUtils speichern
    print(f"✅ Neues JSON gespeichert unter: {processed_path}")

    print("✅ Preprocessing erfolgreich abgeschlossen.")
