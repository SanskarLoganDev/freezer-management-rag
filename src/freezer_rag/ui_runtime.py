"""One synchronized local index per Streamlit process; no conversation state."""

import atexit
import json
from contextlib import ExitStack, closing
from threading import RLock

from freezer_rag.application import ChatSession
from freezer_rag.config import Settings
from freezer_rag.embeddings.ollama import Embeddings
from freezer_rag.ollama import Ollama
from freezer_rag.retrieval.hybrid import Retriever
from freezer_rag.vectorstore.store import IndexStore, store_owner

# Answer model, sampling, and retrieval budgets are deliberately excluded:
# changing them must reuse the same open database and BM25 corpus.
INDEX_FIELDS = (
    "ollama_base_url",
    "embedding_model",
    "embedding_dimensions",
    "embedding_batch_size",
    "embedding_keep_alive",
    "request_timeout_seconds",
    "documents_dir",
    "processed_dir",
    "manifests_dir",
    "qdrant_path",
    "qdrant_collection",
    "chunk_size_chars",
    "chunk_overlap_chars",
)


def index_key(settings: Settings) -> str:
    return settings.model_dump_json(include=set(INDEX_FIELDS))


class SharedIndex:
    def __init__(self, key: str):
        settings = Settings(_env_file=None, **json.loads(key))
        self.lock = RLock()
        self.resources = ExitStack()
        try:
            self.resources.enter_context(store_owner(settings.qdrant_path))
            ollama = self.resources.enter_context(closing(Ollama(settings)))
            self.store = self.resources.enter_context(
                closing(IndexStore(settings, Embeddings(ollama)))
            )
            self.store.open(shared=True)
            self.retriever = Retriever(self.store)
        except BaseException:
            self.resources.close()
            raise
        atexit.register(self.close)

    def session(self, settings: Settings) -> ChatSession:
        return ChatSession(self.store, settings=settings, retriever=self.retriever, lock=self.lock)

    def close(self) -> None:
        with self.lock:
            self.resources.close()
