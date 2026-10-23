"""Bounded evidence prompts and strict citation validation before display."""

import json
import re
import unicodedata
from pathlib import Path

from pydantic import ValidationError

from freezer_rag.config import Settings
from freezer_rag.guardrails import has_override
from freezer_rag.models import Answer, Chunk, GeneratedAnswer, RagError
from freezer_rag.ollama import Ollama
from freezer_rag.retrieval.hybrid import DURATION_QUERY, DURATION_VALUE

PROMPT_PATH = Path(__file__).with_name("system_prompt.txt")


def excerpt(text: str, question: str, byte_limit: int) -> str:
    """Choose complete relevant source lines where possible; never invent a summary."""
    if len(text.encode("utf-8")) <= byte_limit:
        return text
    stop = {"the", "is", "a", "an", "of", "to", "in", "and", "what", "which", "how", "does"}
    terms = set(re.findall(r"[a-z0-9_/-]+", question.lower())) - stop
    # Keep a complete table row together, including its qualifications after a
    # period. Splitting '. | Yes' into a separate fragment obscures state meaning.
    lines = []
    for line in text.splitlines():
        parts = [line] if "|" in line else re.split(r"(?<=[.!?])\s+", line)
        lines.extend(part for part in parts if part.strip())
    ranked = sorted(
        range(len(lines)),
        key=lambda index: (
            bool(DURATION_QUERY.search(question) and DURATION_VALUE.search(lines[index])),
            sum(term in lines[index].lower() for term in terms),
        ),
        reverse=True,
    )
    selected = {}
    remaining = byte_limit
    for index in ranked:
        line = lines[index]
        encoded = line.encode("utf-8")
        if len(encoded) + 1 > remaining:
            if selected:
                continue
            line = encoded[: max(0, remaining - 4)].decode("utf-8", errors="ignore") + "..."
        selected[index] = line
        remaining -= len(line.encode("utf-8")) + 1
        if remaining < 40:
            break
    return "\n".join(selected[index] for index in sorted(selected))


def prompt_cost(messages: list[dict], schema: dict) -> int:
    # UTF-8 bytes are a deliberately conservative upper bound for the supported
    # byte-level tokenizers. Reserve extra space for chat/template/schema framing.
    return (
        sum(len(message["content"].encode("utf-8")) + 24 for message in messages)
        + len(json.dumps(schema).encode("utf-8"))
        + 128
    )


def build_messages(
    settings: Settings,
    question: str,
    chunks: list[Chunk],
    history: list[tuple[str, str]],
    *,
    required_sources: tuple[str, ...] = (),
) -> tuple[list[dict], dict[str, Chunk]]:
    schema = GeneratedAnswer.model_json_schema()
    system = PROMPT_PATH.read_text(encoding="utf-8")
    # Reserve space for schema enums, the single repair instruction, and framing.
    budget = settings.llm_num_ctx - min(128, settings.llm_num_predict) - 512
    messages = [{"role": "system", "content": system}]
    final = {
        "role": "user",
        "content": json.dumps(
            {"question": question, "required_sources": required_sources, "evidence": []}
        ),
    }
    if prompt_cost([*messages, final], schema) > budget:
        raise RagError("Question/instructions exceed the context budget. Shorten the question.")
    # Follow-ups have already been resolved by the planner. Previously generated
    # answers never enter the evidence selector or take space away from sources.
    evidence = []
    sources = {}
    used_chars = 0
    for index, chunk in enumerate(chunks):
        label = f"S{len(sources) + 1}"
        current = {
            "role": "user",
            "content": json.dumps(
                {"question": question, "required_sources": required_sources, "evidence": evidence},
                ensure_ascii=False,
            ),
        }
        available = budget - prompt_cost([*messages, current], schema)
        count = len(chunks) - index
        allowance = min((available - 96 * count) // count, settings.max_context_chars - used_chars)
        if allowance < 100:
            continue
        text = excerpt(chunk.text, question, allowance)
        if has_override(text):
            continue
        content = {"id": label, "source": chunk.source, "text": text}
        candidate = evidence + [content]
        final = {
            "role": "user",
            "content": json.dumps(
                {"question": question, "required_sources": required_sources, "evidence": candidate},
                ensure_ascii=False,
            ),
        }
        if used_chars + len(text) > settings.max_context_chars:
            continue
        if prompt_cost([*messages, final], schema) > budget:
            continue
        evidence = candidate
        sources[label] = chunk.model_copy(update={"text": text})
        used_chars += len(text)
    final = {
        "role": "user",
        "content": json.dumps(
            {"question": question, "required_sources": required_sources, "evidence": evidence},
            ensure_ascii=False,
        ),
    }
    return [*messages, final], sources


def validate_answer(
    content: str, sources: dict[str, Chunk], *, required_sources: tuple[str, ...] = ()
) -> Answer:
    try:
        generated = GeneratedAnswer.model_validate_json(content)
    except ValidationError as exc:
        raise RagError(
            "Model returned an invalid answer structure; no answer was accepted. Retry."
        ) from exc
    if not generated.evidence_sufficient:
        return Answer(
            text="I could not find enough supporting text in the retrieved documents. "
            "Try a more specific document question."
        )
    ids = set(generated.source_ids)
    if not ids.issubset(sources):
        raise RagError(
            "Model returned unknown or inconsistent citations; no answer was accepted. Retry."
        )
    if not ids:
        raise RagError(
            "Model claimed sufficient evidence without citations; no answer was accepted."
        )
    selected = {key: chunk for key, chunk in sources.items() if key in ids}
    if not set(required_sources).issubset({chunk.source for chunk in selected.values()}):
        raise RagError("The selected evidence does not cover all requested documents.")
    # The model selects IDs, never user-visible prose. Valid IDs therefore cannot
    # launder unrelated algorithms or invented facts into accepted output.
    paragraphs = []
    for label, chunk in selected.items():
        if has_override(chunk.text):
            raise RagError("Selected source contains instructions that cannot be used as evidence.")
        text = "".join(
            char
            for char in chunk.text
            if char in "\n\t" or not unicodedata.category(char).startswith("C")
        )
        paragraphs.append(f"{text.strip()} [{label}]")
    return Answer(
        text="From the supplied documents:\n\n" + "\n\n".join(paragraphs),
        sources=selected,
        evidence_sufficient=True,
    )


def generate(
    ollama: Ollama,
    question: str,
    chunks: list[Chunk],
    history: list[tuple[str, str]],
    *,
    required_sources: tuple[str, ...] = (),
) -> Answer:
    if len(required_sources) > 1:
        # One complete top-ranked excerpt per requested document prevents several
        # near-duplicate PDF table fragments crowding out the other contract.
        chunks = [
            next(chunk for chunk in chunks if chunk.source == source)
            for source in required_sources
            if any(chunk.source == source for chunk in chunks)
        ]
    messages, sources = build_messages(
        ollama.settings, question, chunks, history, required_sources=required_sources
    )
    if not sources:
        return Answer(
            text="No usable evidence fits the prompt. Try a more specific question or "
            "increase the context budget."
        )
    schema = GeneratedAnswer.model_json_schema()
    if not set(required_sources).issubset({chunk.source for chunk in sources.values()}):
        return Answer(
            text="The retrieved excerpts do not cover all requested documents. "
            "Try naming the endpoint or section, or increase the evidence budget."
        )
    schema["properties"]["source_ids"]["items"]["enum"] = list(sources)
    if len(required_sources) > 1:
        messages[0]["content"] += "\nFor this comparison select all of " + ", ".join(sources) + "."
    for attempt in range(2):
        content = ollama.chat(messages, schema, num_predict=128)
        try:
            answer = validate_answer(content, sources, required_sources=required_sources)
            if answer.evidence_sufficient or attempt:
                return answer
        except RagError:
            if attempt:
                return Answer(
                    text="I could not produce a source-verified answer. "
                    "Please rephrase the document question."
                )
        # One bounded repair uses the same model and original evidence, without
        # replaying a rejected draft or its potentially injected text.
        messages[0] = {
            "role": "system",
            "content": messages[0]["content"]
            + "\nSelect valid IDs, cover required sources, and correct false premises. "
            "Use false only when no supplied excerpt answers the question. "
            + "Available sources: "
            + "; ".join(f"{label}={chunk.source}" for label, chunk in sources.items()),
        }
    raise AssertionError("Unreachable selection retry state")
