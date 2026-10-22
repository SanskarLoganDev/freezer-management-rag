"""Shared terminal/UI application service; no HTTP server required."""

from contextlib import closing
from threading import RLock

from freezer_rag.config import Settings
from freezer_rag.generation.answers import generate
from freezer_rag.models import Answer, Chunk, RagError
from freezer_rag.ollama import Ollama
from freezer_rag.retrieval.hybrid import Retriever, tokenize
from freezer_rag.vectorstore.store import IndexStore


class ChatSession:
    def __init__(
        self,
        store: IndexStore,
        *,
        settings: Settings | None = None,
        retriever: Retriever | None = None,
        lock=None,
    ):
        self.store = store
        self.settings = settings or store.settings
        self.retriever = retriever or Retriever(store)
        self.lock = lock or RLock()
        self.history: list[tuple[str, str]] = []
        self.last_answer: Answer | None = None

    def clear(self) -> None:
        with self.lock:
            self.history.clear()
            self.last_answer = None

    def search(self, question: str) -> list[Chunk]:
        if not question.strip():
            raise RagError("Please enter a question")
        if len(question) > self.settings.max_question_chars:
            raise RagError(f"Question exceeds {self.settings.max_question_chars} characters")
        # Recent user questions supply referents for short follow-ups. They do not
        # become evidence and prior answers never enter the searchable corpus.
        query = question
        # Do not drown an explicit new subject (e.g. reservation duration) in
        # terms from the prior question. Expand only genuinely terse follow-ups.
        if self.history and len(set(tokenize(question))) < 3:
            query = self.history[-1][0][:240] + "\nFollow-up: " + question
        with self.lock:
            return self.retriever.search(query, self.settings)

    def ask(self, question: str) -> Answer:
        with self.lock:
            chunks = self.search(question)
            if self.settings is self.store.settings:
                answer = generate(self.store.embeddings.ollama, question, chunks, self.history)
            else:
                # UI settings are private to this conversation. Never mutate the
                # cached index's settings or share another session's history/client.
                with closing(Ollama(self.settings)) as ollama:
                    answer = generate(ollama, question, chunks, self.history)
            self.last_answer = answer
            self.history.append((question, answer.text))
            keep = self.settings.max_history_turns
            self.history = self.history[-keep:] if keep else []
            return answer
