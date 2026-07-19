from src import agents, memory, orchestration, providers


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def retrieve_documents(self, query, history):
        self.calls.append((query, history))
        return [
            {
                "chunk_id": "doc:000000:paragraph:part-0000",
                "text": "retrieved context",
                "metadata": {
                    "document_title": "Document",
                    "page_number": 1,
                    "source_type": "paragraph",
                },
                "distance": 0.0,
            }
        ]


class FakeGenerator:
    def __init__(self):
        self.calls = []

    def generate_answer(self, query, records, history, *, session_id):
        self.calls.append((query, records, history, session_id))
        return providers.contracts.GenerationResult(
            answer=f"answer to {query}",
            provider_id="fake",
            model_id="fake-model",
            usage=providers.contracts.GenerationUsage(),
        )


def test_chat_id_is_explicit_and_histories_do_not_cross_sessions():
    conversation_store = memory.in_memory.InMemoryConversationStore(max_history=10)
    retriever = FakeRetriever()
    generator = FakeGenerator()
    chatbot = orchestration.rag.RAGChatbot(
        retriever_agent=retriever,
        generator_agent=generator,
        memory_agent=agents.memory.MemoryAgent(conversation_store),
    )

    assert chatbot.process_user_input("first", chat_id="session-a").answer == (
        "answer to first"
    )
    assert chatbot.process_user_input("other", chat_id="session-b").answer == (
        "answer to other"
    )
    assert chatbot.process_user_input("follow-up", chat_id="session-a").answer == (
        "answer to follow-up"
    )

    assert retriever.calls[0] == ("first", [])
    assert retriever.calls[1] == ("other", [])
    assert retriever.calls[2][0] == "follow-up"
    assert retriever.calls[2][1] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer to first"},
    ]
    assert conversation_store.get_history("session-b") == [
        {"role": "user", "content": "other"},
        {"role": "assistant", "content": "answer to other"},
    ]
