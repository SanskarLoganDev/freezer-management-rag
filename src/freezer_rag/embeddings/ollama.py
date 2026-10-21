"""Model-specific retrieval formatting and validated, normalized embeddings."""

import math

from freezer_rag.models import RagError
from freezer_rag.ollama import Ollama


class Embeddings:
    def __init__(self, ollama: Ollama):
        self.ollama = ollama
        self.settings = ollama.settings
        self.model = self.settings.embedding_model
        info = ollama.show(self.model)
        self.digest = ollama.model(self.model)["digest"]
        self.strategy = "nomic-search-v1" if self.model.startswith("nomic") else "qwen-instruct-v1"
        self.template = str(info.get("template", ""))
        # Some installed templates apply the retrieval prefix themselves.
        self.template_prefixes = "search_document:" in self.template
        self.strategy += "-template" if self.template_prefixes else "-application"
        lengths = [
            int(value)
            for key, value in info.get("model_info", {}).items()
            if key.endswith(".context_length") and isinstance(value, (int, float))
        ]
        self.input_limit = min(lengths) if lengths else 2048

    def format(self, text: str, query: bool) -> str:
        if self.model.startswith("nomic"):
            if self.template_prefixes:
                if query:
                    raise RagError(
                        "The installed embedding template forces search_document formatting. "
                        "Use a template that permits distinct document and query prefixes."
                    )
                return text
            return ("search_query: " if query else "search_document: ") + text
        if query and "Instruct:" not in self.template:
            return (
                "Instruct: Retrieve passages answering the freezer-management question.\nQuery: "
                + text
            )
        return text

    def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        formatted = [self.format(text, query) for text in texts]
        # For these byte-level tokenizers, UTF-8 bytes conservatively bound input tokens.
        if any(len(text.encode("utf-8")) + 32 > self.input_limit for text in formatted):
            raise RagError(
                "Embedding input exceeds the conservative model input budget. "
                "Reduce RAG_CHUNK_SIZE_CHARS and re-ingest, or shorten the question."
            )
        result = self.ollama.request(
            "POST",
            "/api/embed",
            {
                "model": self.model,
                "input": formatted,
                "truncate": False,
                "keep_alive": self.settings.embedding_keep_alive,
            },
            retries=2,
        )
        vectors = result.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise RagError("Ollama returned the wrong number of embedding vectors")
        normalized = []
        for vector in vectors:
            if not isinstance(vector, list) or len(vector) != self.settings.embedding_dimensions:
                raise RagError(
                    "Embedding dimension mismatch; check the selected model and configuration"
                )
            if not all(
                isinstance(value, (int, float)) and math.isfinite(value) for value in vector
            ):
                raise RagError("Ollama returned a non-finite or non-numeric embedding")
            magnitude = math.sqrt(sum(value * value for value in vector))
            if not math.isfinite(magnitude) or magnitude == 0:
                raise RagError("Ollama returned an invalid zero/overflowed embedding")
            normalized.append([value / magnitude for value in vector])
        return normalized
