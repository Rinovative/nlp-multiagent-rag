import hashlib

import pytest

from src import ingestion

UniversalPDFLoader = ingestion.loader.UniversalPDFLoader


class FakePage:
    def __init__(self, text):
        self._text = text
        self.extract_text_calls = 0
        self.chars = [
            {
                "text": character,
                "top": 100.0,
                "size": 12.0,
                "fontname": "TestFont",
            }
            for character in text
        ]
        self.annots = [{"uri": "https://example.com/source"}]

    def extract_text(self):
        self.extract_text_calls += 1
        return self._text

    def extract_tables(self):
        return [[["A", "B"], ["1", "2"]]]


class FakePDF:
    def __init__(self, pages):
        self.pages = pages
        self.metadata = {
            "Title": "Tracked fixture",
            "Author": "Test author",
            "Subject": "Testing",
            "Producer": "Fake producer",
            "CreationDate": "today",
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def test_loader_processes_upload_bytes_without_a_temporary_file(monkeypatch):
    content = b"%PDF deterministic bytes"
    page = FakePage("This is enough English text for language detection.")
    monkeypatch.setattr(
        "src.ingestion.loader.pdfplumber.open", lambda _source: FakePDF([page])
    )

    result = UniversalPDFLoader().load_pdf(
        content, file_name="same-name.pdf", extract_tables=True
    )

    assert result["metadata"]["file_name"] == "same-name.pdf"
    assert result["metadata"]["file_path"] is None
    assert result["metadata"]["file_size"] == len(content)
    assert result["metadata"]["file_hash"] == hashlib.sha256(content).hexdigest()
    assert result["metadata"]["num_pages"] == 1
    assert result["pages"][0]["page"] == 1
    assert result["pages"][0]["paragraphs"][0]["text"].startswith("This is")
    assert result["tables"][0]["table"] == [["A", "B"], ["1", "2"]]
    assert result["metadata"]["all_links"] == ["https://example.com/source"]
    assert page.extract_text_calls == 1


def test_loader_rejects_empty_upload_bytes():
    try:
        UniversalPDFLoader().load_pdf(b"", file_name="empty.pdf")
    except ValueError as exc:
        assert "must not be empty" in str(exc)
    else:
        raise AssertionError("Expected empty PDF content to be rejected")


def test_multiline_paragraph_metadata_is_final_and_deterministic():
    page = FakePage("")
    page.chars = [
        *[
            {"text": character, "top": 100.0, "size": 12.0, "fontname": "ZFont"}
            for character in "First"
        ],
        *[
            {"text": character, "top": 110.0, "size": 12.0, "fontname": "AFont"}
            for character in "Second"
        ],
    ]

    paragraphs, assigned_links = UniversalPDFLoader()._extract_paragraphs_with_fonts(
        page,
        all_known_links={"https://z.example", "https://a.example"},
    )

    assert assigned_links == set()
    assert paragraphs == [
        {
            "text": "First Second",
            "font_size": 12.0,
            "font_names": ["AFont", "ZFont"],
            "text_hash": hashlib.sha256(b"First Second").hexdigest(),
            "y_position": 100.0,
            "links": [],
        }
    ]


def test_processor_translates_unexpected_parser_failure_without_leaking_detail():
    class FailingLoader:
        def load_pdf(self, *_args, **_kwargs):
            raise RuntimeError("sensitive parser path")

    processor = ingestion.processor.DocumentProcessor(
        faiss_store=object(),
        embedding_provider=object(),
        loader=FailingLoader(),
    )

    with pytest.raises(ingestion.processor.DocumentProcessingError) as captured:
        processor.prepare_bytes(b"not a pdf", file_name="upload.pdf")

    assert "upload.pdf" in str(captured.value)
    assert "sensitive parser path" not in str(captured.value)
