import copy

import pytest

from src import ingestion

InvalidChunkError = ingestion.chunker.InvalidChunkError
PDFChunker = ingestion.chunker.PDFChunker


def document_with_paragraphs(*texts):
    paragraphs = [
        {
            "text": text,
            "font_size": 12.0,
            "font_names": ["Test"],
            "text_hash": f"hash-{index}",
            "y_position": 100.0 + index,
            "heading_level": 0,
            "is_type": "normal",
            "links": [],
        }
        for index, text in enumerate(texts)
    ]
    return {
        "metadata": {
            "document_title": "Deterministic document",
            "file_name": "same.pdf",
            "file_hash": "a" * 64,
            "document_language": "en",
            "tables": [],
        },
        "pages": [{"page": 1, "paragraphs": paragraphs}],
    }


def test_overlap_and_length_use_characters():
    chunker = PDFChunker(max_chunk_length=10, overlap_length=3)
    parts = chunker.split_text("abcdefghijklmnop")

    assert parts == ["abcdefghij", "hijklmnop"]
    assert parts[0][-3:] == parts[1][:3]
    assert all(len(part) <= 10 for part in parts)


@pytest.mark.parametrize(
    ("length", "expected_parts"), [(9, 1), (10, 1), (11, 2), (25, 3)]
)
def test_boundary_lengths_respect_maximum(length, expected_parts):
    chunker = PDFChunker(max_chunk_length=10, overlap_length=2)
    parts = chunker.split_text("x" * length)
    assert len(parts) == expected_parts
    assert all(0 < len(part) <= 10 for part in parts)


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError, match="0 <= overlap_length"):
        PDFChunker(max_chunk_length=10, overlap_length=10)
    with pytest.raises(ValueError, match="0 <= overlap_length"):
        PDFChunker(max_chunk_length=10, overlap_length=-1)


def test_split_parts_have_unique_deterministic_ids_and_metadata():
    document = document_with_paragraphs("abcdefghijklmnopqrstuvwxyz")
    chunker = PDFChunker(max_chunk_length=10, overlap_length=2)

    first = chunker.chunk_document(copy.deepcopy(document))
    second = chunker.chunk_document(copy.deepcopy(document))

    assert first == second
    assert len(first) == 3
    assert len({chunk["chunk_id"] for chunk in first}) == 3
    assert [chunk["metadata"]["part_index"] for chunk in first] == [0, 1, 2]
    assert all(chunk["metadata"]["part_count"] == 3 for chunk in first)
    assert all(chunk["metadata"]["length_unit"] == "characters" for chunk in first)
    assert all(isinstance(chunk["metadata"], dict) for chunk in first)
    assert all(len(chunk["text"]) <= 10 for chunk in first)


def test_repeated_equal_paragraphs_keep_distinct_positions():
    chunks = PDFChunker(max_chunk_length=100, overlap_length=10).chunk_document(
        document_with_paragraphs("Repeated text", "Repeated text")
    )

    assert [chunk["metadata"]["paragraph_index"] for chunk in chunks] == [0, 1]
    assert chunks[0]["chunk_id"] != chunks[1]["chunk_id"]
    assert [chunk["metadata"]["chunk_sequence"] for chunk in chunks] == [0, 1]


def test_source_order_and_structure_types_are_stable():
    document = document_with_paragraphs("Heading", "Paragraph", "Caption")
    document["pages"][0]["paragraphs"][0]["heading_level"] = 1
    document["pages"][0]["paragraphs"][0]["is_type"] = "heading"
    document["pages"][0]["paragraphs"][2]["is_type"] = "caption"
    document["metadata"]["tables"] = [{"page": 1, "table": [["A", "B"], ["1", "2"]]}]

    chunks = PDFChunker(max_chunk_length=100, overlap_length=10).chunk_document(
        document
    )

    assert [chunk["metadata"]["source_type"] for chunk in chunks] == [
        "heading",
        "paragraph",
        "caption",
        "table",
    ]
    assert [chunk["metadata"]["source_sequence"] for chunk in chunks] == [0, 1, 2, 3]
    assert chunks[-1]["metadata"]["table_index"] == 0
    assert chunks[-1]["text"] == "A | B\n1 | 2"


def test_invalid_chunk_fails_early():
    with pytest.raises(InvalidChunkError, match="metadata must be a dictionary"):
        PDFChunker.validate_chunk(
            {"chunk_id": "bad", "text": "content", "metadata": "not-a-dict"}
        )


def test_missing_file_hash_gets_stable_content_hash():
    document = document_with_paragraphs("Stable content")
    del document["metadata"]["file_hash"]
    chunker = PDFChunker(max_chunk_length=100, overlap_length=10)

    first = chunker.chunk_document(copy.deepcopy(document))
    second = chunker.chunk_document(copy.deepcopy(document))

    assert first[0]["metadata"]["document_id"] == second[0]["metadata"]["document_id"]
    assert len(first[0]["metadata"]["document_id"]) == 64
