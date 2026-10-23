"""Shared terminal/UI application service; no HTTP server required."""

from contextlib import closing
from threading import RLock

from freezer_rag.config import Settings
from freezer_rag.generation.answers import generate
from freezer_rag.guardrails import has_override, plan_request, refusal
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

    def search(self, question: str, *, required_sources: tuple[str, ...] = ()) -> list[Chunk]:
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
            return self.retriever.search(query, self.settings, required_sources=required_sources)

    def _answer(self, ollama: Ollama, question: str) -> tuple[Answer, str | None]:
        plan = plan_request(ollama, question, self.history)
        if plan.intent != "document":
            return refusal(plan.intent), None
        required = tuple(dict.fromkeys(plan.sources))
        if len(required) > self.settings.final_top_k:
            raise RagError(
                "This comparison needs final_top_k at least as large as its source count."
            )
        # The planner resolves follow-ups from accepted questions. Raw mixed
        # instructions, false premises, and old answers never become retrieval facts.
        chunks = self.retriever.search(plan.query, self.settings, required_sources=required)
        answer = generate(ollama, plan.query, chunks, [], required_sources=required)
        return answer, plan.query

    def ask(self, question: str) -> Answer:
        with self.lock:
            if not question.strip():
                raise RagError("Please enter a question")
            if len(question) > self.settings.max_question_chars:
                raise RagError(f"Question exceeds {self.settings.max_question_chars} characters")
            if has_override(question):
                self.last_answer = refusal("override")
                return self.last_answer
            if self.settings is self.store.settings:
                answer, query = self._answer(self.store.embeddings.ollama, question)
            else:
                # UI settings are private to this conversation. Never mutate the
                # cached index's settings or share another session's history/client.
                with closing(Ollama(self.settings)) as ollama:
                    answer, query = self._answer(ollama, question)
            self.last_answer = answer
            if answer.evidence_sufficient and query:
                self.history.append((query, answer.text))
            keep = self.settings.max_history_turns
            self.history = self.history[-keep:] if keep else []
            return answer
