"""Exact-identifier aware BM25 plus dense retrieval, combined by rank."""

import re

from rank_bm25 import BM25Okapi

from freezer_rag.ingestion.chunking import normalize
from freezer_rag.models import Chunk
from freezer_rag.vectorstore.store import IndexStore

DURATION_QUERY = re.compile(r"\b(how long|duration|expires?|expiry|expiration)\b", re.I)
DURATION_VALUE = re.compile(
    r"\b(?:\d+|one|two|three|four|five)\s*[- ]?\s*(?:minutes?|hours?|days?)\b", re.I
)

STOP_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "by",
    "at",
    "as",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "what",
    "which",
    "who",
    "how",
    "when",
    "where",
    "why",
    "does",
    "do",
    "did",
    "can",
    "could",
    "would",
    "should",
    "with",
    "from",
    "between",
    "about",
    "me",
    "my",
    "i",
}


def tokenize(text: str) -> list[str]:
    text = text.lower()
    # Preserve endpoint paths and IDs as well as their individual words.
    identifiers = re.findall(r"/[a-z0-9_/{}/.-]+|\b[a-z]+-\d+\b", text)
    words = re.findall(r"[a-z0-9]+(?:_[a-z0-9]+)*", text)
    return identifiers + [word for word in words if word not in STOP_WORDS]


class Retriever:
    def __init__(self, store: IndexStore):
        self.store = store
        self.chunks = {chunk.id: chunk for chunk in store.chunks}
        self.ids = list(self.chunks)
        self.bm25 = BM25Okapi([tokenize(chunk.normalized_text) for chunk in store.chunks])

    def search(self, question: str) -> list[Chunk]:
        settings = self.store.settings
        vector = self.store.embeddings.embed([question], query=True)[0]
        dense = self.store.dense(vector, settings.dense_top_k)
        terms = tokenize(question)
        # Duration questions often omit the vocabulary used in the specifications.
        # Expand search vocabulary only; deadlines still come entirely from sources.
        duration_question = bool(DURATION_QUERY.search(question))
        if duration_question:
            terms += ["expiry", "expiration", "hours", "days"]
        scores = self.bm25.get_scores(terms)
        lexical = [
            self.ids[index]
            for index in sorted(
                range(len(scores)), key=lambda index: float(scores[index]), reverse=True
            )
            if scores[index] > 0
        ][: settings.lexical_top_k]
        fused: dict[str, float] = {}
        for ranking in (dense, lexical):
            for rank, chunk_id in enumerate(ranking, 1):
                fused[chunk_id] = fused.get(chunk_id, 0) + 1 / (60 + rank)
        selected = []
        seen: list[set[str]] = []
        # For duration questions, prefer passages containing a stated duration over
        # captions or generic expiry mentions. Values remain source evidence, never rules.
        ranked = sorted(
            fused,
            key=lambda item: (
                duration_question
                and bool(DURATION_VALUE.search(self.chunks[item].normalized_text)),
                fused[item],
            ),
            reverse=True,
        )
        for chunk_id in ranked:
            chunk = self.chunks[chunk_id]
            tokens = set(tokenize(normalize(chunk.text)))
            if any(len(tokens & prior) / max(1, len(tokens | prior)) > 0.85 for prior in seen):
                continue
            selected.append(chunk)
            seen.append(tokens)
            if len(selected) >= settings.final_top_k:
                break
        return selected
