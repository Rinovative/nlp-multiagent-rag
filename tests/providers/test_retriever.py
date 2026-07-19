import pytest

from src import agents


class RecordingEmbedder:
    model_id = "test-model"
    dimension = 2

    def __init__(self):
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        return [1.0, 0.0]


class RecordingStore:
    record_count = 1
    embedding_model = "test-model"
    dimension = 2

    def __init__(self):
        self.searches = []

    def search(self, embedding, *, k):
        self.searches.append((embedding, k))
        return [{"chunk_id": "one", "text": "result", "metadata": {}}]


def test_retriever_keeps_current_question_first_and_bounds_history():
    store = RecordingStore()
    embedder = RecordingEmbedder()
    retriever = agents.retriever.RetrieverAgent(
        store,
        embedder,
        top_k=3,
        max_query_characters=100,
    )

    results = retriever.retrieve_documents(
        "current question",
        [
            {"role": "user", "content": "old " * 30},
            {"role": "assistant", "content": "ignored answer"},
            {"role": "user", "content": "recent question"},
        ],
    )

    embedded = embedder.queries[0]
    assert embedded.startswith("Current question:\ncurrent question")
    assert "recent question" in embedded
    assert "ignored answer" not in embedded
    assert len(embedded) <= 100
    assert store.searches == [([1.0, 0.0], 3)]
    assert results[0]["chunk_id"] == "one"


def test_retriever_rejects_oversized_question_before_embedding():
    retriever = agents.retriever.RetrieverAgent(
        RecordingStore(),
        RecordingEmbedder(),
        max_query_characters=30,
    )

    with pytest.raises(agents.retriever.RetrievalValidationError):
        retriever.retrieve_documents("x" * 30, [])


@pytest.mark.parametrize(
    ("store_model", "store_dimension"),
    [("different-model", 2), ("test-model", 3)],
)
def test_retriever_rejects_incompatible_vector_store(
    store_model,
    store_dimension,
):
    store = RecordingStore()
    store.embedding_model = store_model
    store.dimension = store_dimension

    with pytest.raises(agents.retriever.RetrievalConfigurationError):
        agents.retriever.RetrieverAgent(store, RecordingEmbedder())
