from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
from src.agents.retriever_agent import RetrieverAgent
from src.agents.generator_agent import GeneratorAgent
from src.agents.memory_agent import MemoryAgent
from src.vectorstore.faiss_store import FAISSStore
from src.memory.memory import MemoryStorage
import openai


# Definiere den State, der für die Nachrichten im Graphen zuständig ist
class State(TypedDict):
    messages: Annotated[list, add_messages]


class RAGChatbot:
    def __init__(
        self, faiss_store: FAISSStore, client: openai, memory_storage: MemoryStorage
    ):
        """
        Initialisiert den RAGChatbot mit einem FAISS-Store, einem OpenAI Client und einem API-Key.
        """
        self.client = client

        # Initialisiere den Graphen
        self.graph_builder = StateGraph(State)

        # Initialisiere die Agenten
        self.memory_agent = MemoryAgent(memory_storage)
        self.retriever_agent = RetrieverAgent(faiss_store, self.client)
        self.generator_agent = GeneratorAgent(self.client)

        # Füge Funktionen hinzu
        self.graph_builder.add_node("get_memory", self.get_memory_function)
        self.graph_builder.add_node("retriever", self.retriever_function)
        self.graph_builder.add_node("generator", self.generator_function)
        self.graph_builder.add_node("set_memory", self.set_memory_function)

        # Definiere den Ablauf
        self.graph_builder.add_edge(
            START, "get_memory"
        )  # Der Startknoten führt zuerst zum MemoryAgent
        self.graph_builder.add_edge(
            "get_memory", "retriever"
        )  # MemoryAgent geht zum Retriever
        self.graph_builder.add_edge(
            "retriever", "generator"
        )  # Retriever geht zum Generator
        self.graph_builder.add_edge(
            "generator", "set_memory"
        )  # Generator geht zurück zum MemoryAgent, um den Kontext zu aktualisieren
        self.graph_builder.add_edge("set_memory", END)  # Generator endet den Prozess

        # Kompiliere den Graphen
        self.graph = self.graph_builder.compile()

    def get_memory_function(self, state: State):
        """
        Funktion für den Memory-Agenten, der den Kontext basierend auf dem Chatverlauf und den Benutzerpräferenzen abruft.
        """
        # Zugriff auf die letzte Nachricht
        message = state["messages"][-1].content  # Inhalt der letzten Nachricht
        if "chat_id" not in self.__dict__:
            self.chat_id = state["messages"][-1].id

        # Abrufen des gespeicherten Kontexts (leer beim ersten Mal)
        context = self.memory_agent.get_memory(self.chat_id)
        self.memory_agent.add_to_memory(self.chat_id, "user", message)

        return {"messages": [{"role": "system", "content": context}]}

    def retriever_function(self, state: State):
        """
        Funktion für den Retriever-Agenten, der Dokumente basierend auf der Anfrage abruft.
        """
        # Zugriff auf die Benutzeranfrage und die Chat-ID
        user_message = state["messages"][
            -2
        ].content  # Die Benutzeranfrage (vorherige Nachricht)
        memory = state["messages"][-1].content  # Memory

        # Abruf des Kontexts durch den RetrieverAgent basierend auf der Benutzeranfrage
        retrieved_context = self.retriever_agent.retrieve_documents(
            user_message, memory
        )

        # Rückgabe des abgerufenen Kontexts für die nächste Phase (z. B. GeneratorAgent)
        return {"messages": [{"role": "system", "content": retrieved_context}]}

    def generator_function(self, state: State):
        """
        Funktion für den Generator-Agenten, der die Antwort basierend auf dem Kontext generiert.
        """
        # Zugriff auf den abgerufenen Kontext
        context = state["messages"][-1].content
        memory = state["messages"][-2].content
        user_message = state["messages"][-3].content

        # Generiere die Antwort basierend auf dem kombinierten Kontext
        answer = self.generator_agent.generate_answer(user_message, context, memory)

        return {"messages": [{"role": "assistant", "content": answer}]}

    def set_memory_function(self, state: State):
        """
        Funktion für den Memory-Agenten, der den Kontext basierend auf dem Chatverlauf und den Benutzerpräferenzen abruft.
        """
        # Zugriff auf die letzte Nachricht
        answer = state["messages"][-1].content
        context = state["messages"][-2].content

        self.memory_agent.add_to_memory(self.chat_id, "context", context)
        self.memory_agent.add_to_memory(self.chat_id, "assistant", answer)

        return {"messages": [{"role": "system", "content": answer}]}

    def process_user_input(self, user_input: str):
        """
        Funktion, um die Benutzereingabe zu verarbeiten und durch den Graphen zu streamen.
        """
        response = None  # Initialisiere die Antwortvariable

        # Durchlaufe den Graphen und verarbeite die Benutzereingabe
        for event in self.graph.stream(
            {"messages": [{"role": "user", "content": user_input}]}
        ):
            for value in event.values():
                response = value["messages"][-1]["content"]

        return response  # Gebe die Antwort zurück
