import copy

from src import ingestion

PdfPreprocessor = ingestion.preprocessing.PdfPreprocessor


def paragraph(text, *, size=12.0, y=200.0):
    return {
        "text": text,
        "font_size": size,
        "font_names": ["TestFont"],
        "text_hash": text,
        "y_position": y,
        "links": [],
    }


def test_preprocessor_produces_chunker_ready_metadata_deterministically():
    document = {
        "metadata": {
            "document_title": None,
            "file_name": "fixture.pdf",
            "file_hash": "c" * 64,
        },
        "pages": [
            {
                "page": 1,
                "paragraphs": [
                    paragraph("Document title", size=18.0, y=120.0),
                    paragraph("First body paragraph.", y=220.0),
                ],
            },
            {
                "page": 2,
                "paragraphs": [paragraph("Second body paragraph.", y=220.0)],
            },
        ],
        "tables": [],
    }

    first, removed = PdfPreprocessor(copy.deepcopy(document)).run_preprocessing()
    second, _ = PdfPreprocessor(copy.deepcopy(document)).run_preprocessing()

    assert first == second
    assert first["metadata"]["document_title"] == "Document title"
    assert first["pages"][0]["paragraphs"][0]["is_type"] == "heading"
    assert isinstance(first["metadata"]["font_size_stats"], dict)
    assert isinstance(first["metadata"]["headings"], list)
    assert isinstance(first["metadata"]["tables"], list)
    assert isinstance(first["metadata"]["captions"], list)
    assert isinstance(first["metadata"]["pseudo_tables"], list)
    assert "tables" not in first
    assert all(
        isinstance(item.get("heading_level"), int)
        and isinstance(item.get("is_type"), str)
        for page in first["pages"]
        for item in page["paragraphs"]
    )
    assert set(removed) == {
        "removed_headers_candidates",
        "removed_footers_candidates",
        "removed_headers_fallback",
        "removed_footers_fallback",
    }


def test_missing_font_statistics_produce_safe_empty_candidates():
    document = {
        "metadata": {"file_hash": "f" * 64},
        "pages": [
            {
                "page": 1,
                "paragraphs": [paragraph("Text without a font size", size=None)],
            }
        ],
        "tables": [],
    }

    processed, removed = PdfPreprocessor(document).run_preprocessing()

    assert processed["metadata"]["main_font_size"] is None
    assert processed["metadata"]["header_footer_candidate_sizes"] == []
    assert processed["pages"][0]["paragraphs"][0]["is_type"] == "normal"
    assert removed == {}


def test_only_repeated_header_candidates_are_removed():
    document = {
        "metadata": {"file_hash": "d" * 64},
        "pages": [
            {
                "page": 1,
                "paragraphs": [
                    paragraph("Repeated header", size=10.0, y=50.0),
                    paragraph("First body", y=220.0),
                    paragraph("First body continued", y=260.0),
                ],
            },
            {
                "page": 2,
                "paragraphs": [
                    paragraph("Repeated header", size=10.0, y=50.0),
                    paragraph("Second body", y=220.0),
                    paragraph("Second body continued", y=260.0),
                ],
            },
            {
                "page": 3,
                "paragraphs": [
                    paragraph("Section-specific note", size=10.0, y=50.0),
                    paragraph("Third body", y=220.0),
                    paragraph("Third body continued", y=260.0),
                ],
            },
        ],
        "tables": [],
    }

    processed, removed = PdfPreprocessor(document).run_preprocessing()
    remaining = [
        item["text"] for page in processed["pages"] for item in page["paragraphs"]
    ]

    assert "Repeated header" not in remaining
    assert "Section-specific note" in remaining
    assert removed["removed_headers_candidates"] == [
        "Repeated header",
        "Repeated header",
    ]


def test_pseudo_table_lookahead_does_not_consume_rest_of_page():
    document = {
        "metadata": {"file_hash": "e" * 64},
        "pages": [
            {
                "page": 1,
                "paragraphs": [
                    paragraph("Contents .... 1"),
                    paragraph("Continuation one"),
                    paragraph("Continuation two"),
                    paragraph("Continuation three"),
                    paragraph("Ordinary paragraph"),
                ],
            }
        ],
        "tables": [],
    }

    processed, _ = PdfPreprocessor(document).run_preprocessing()
    paragraphs = processed["pages"][0]["paragraphs"]

    assert [item["is_type"] for item in paragraphs[:4]] == ["pseudo_table"] * 4
    assert paragraphs[4]["is_type"] == "normal"
