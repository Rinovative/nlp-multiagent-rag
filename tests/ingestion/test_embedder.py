from __future__ import annotations

import numpy as np
import pytest

from src import embeddings, ingestion


class FakeSentenceTransformer:
    def __init__(
        self,
        *,
        dimension: int = 3,
        invalid_shape: bool = False,
        non_finite: bool = False,
    ):
        self.dimension = dimension
        self.invalid_shape = invalid_shape
        self.non_finite = non_finite
        self.calls: list[dict] = []

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode(self, texts, **kwargs):
        self.calls.append({"texts": list(texts), **kwargs})
        dimension = self.dimension + 1 if self.invalid_shape else self.dimension
        result = np.asarray(
            [[float(index + 1)] * dimension for index, _ in enumerate(texts)],
            dtype=np.float32,
        )
        if self.non_finite:
            result[0, 0] = np.nan
        return result


def valid_chunks():
    document = {
        "metadata": {
            "document_title": "Embedding test",
            "file_hash": "b" * 64,
            "tables": [],
        },
        "pages": [
            {
                "page": 1,
                "paragraphs": [
                    {"text": "first", "heading_level": 0, "is_type": "normal"},
                    {"text": "second", "heading_level": 0, "is_type": "normal"},
                ],
            }
        ],
    }
    return ingestion.chunker.PDFChunker(
        max_chunk_length=100, overlap_length=10
    ).chunk_document(document)


def test_local_embedder_is_lazy_batched_normalized_and_prefix_aware():
    model = FakeSentenceTransformer()
    factory_calls: list[str] = []

    def factory(model_id: str):
        factory_calls.append(model_id)
        return model

    provider = embeddings.sentence_transformer.SentenceTransformerEmbeddingProvider(
        model_id="intfloat/multilingual-e5-small",
        dimension=3,
        batch_size=2,
        model_factory=factory,
    )
    assert factory_calls == []

    document_vectors = provider.embed_documents(["alpha", "beta"])
    query_vector = provider.embed_query("alpha")

    assert factory_calls == ["intfloat/multilingual-e5-small"]
    assert model.calls[0]["texts"] == ["passage: alpha", "passage: beta"]
    assert model.calls[1]["texts"] == ["query: alpha"]
    assert model.calls[0]["batch_size"] == 2
    assert model.calls[0]["normalize_embeddings"] is True
    assert document_vectors[0] == [1.0, 1.0, 1.0]
    assert query_vector == [1.0, 1.0, 1.0]


def test_chunk_embedding_preserves_schema_without_mutation():
    model = FakeSentenceTransformer()
    provider = embeddings.sentence_transformer.SentenceTransformerEmbeddingProvider(
        model_id="test-model",
        dimension=3,
        use_e5_prefixes=False,
        model_factory=lambda _model_id: model,
    )
    chunks = valid_chunks()

    result = embeddings.chunks.embed_chunks(chunks, provider)

    assert [item["chunk_id"] for item in result] == [
        chunk["chunk_id"] for chunk in chunks
    ]
    assert result[0]["metadata"] == chunks[0]["metadata"]
    assert result[0]["metadata"] is not chunks[0]["metadata"]
    assert model.calls[0]["texts"] == [chunk["text"] for chunk in chunks]


def test_invalid_chunk_is_rejected_before_model_loading():
    factory_calls = 0

    def factory(_model_id: str):
        nonlocal factory_calls
        factory_calls += 1
        return FakeSentenceTransformer()

    provider = embeddings.sentence_transformer.SentenceTransformerEmbeddingProvider(
        model_id="test-model", dimension=3, model_factory=factory
    )
    with pytest.raises(embeddings.contracts.EmbeddingError):
        embeddings.chunks.embed_chunks(
            [{"chunk_id": "bad", "text": "text", "metadata": "invalid"}],
            provider,
        )
    assert factory_calls == 0


def test_model_dimension_and_response_shape_are_rejected():
    wrong_dimension = (
        embeddings.sentence_transformer.SentenceTransformerEmbeddingProvider(
            model_id="test-model",
            dimension=3,
            model_factory=lambda _model_id: FakeSentenceTransformer(dimension=4),
        )
    )
    with pytest.raises(embeddings.contracts.EmbeddingError):
        wrong_dimension.embed_query("query")

    wrong_shape = embeddings.sentence_transformer.SentenceTransformerEmbeddingProvider(
        model_id="test-model",
        dimension=3,
        model_factory=lambda _model_id: FakeSentenceTransformer(invalid_shape=True),
    )
    with pytest.raises(embeddings.contracts.EmbeddingError):
        wrong_shape.embed_query("query")


def test_non_finite_embedding_values_are_rejected():
    provider = embeddings.sentence_transformer.SentenceTransformerEmbeddingProvider(
        model_id="test-model",
        dimension=3,
        model_factory=lambda _model_id: FakeSentenceTransformer(non_finite=True),
    )

    with pytest.raises(embeddings.contracts.EmbeddingError, match="non-finite"):
        provider.embed_query("query")


def test_empty_document_input_avoids_model_loading():
    provider = embeddings.sentence_transformer.SentenceTransformerEmbeddingProvider(
        model_id="test-model",
        dimension=3,
        model_factory=lambda _model_id: pytest.fail("model should remain unloaded"),
    )
    assert provider.embed_documents([]) == []
