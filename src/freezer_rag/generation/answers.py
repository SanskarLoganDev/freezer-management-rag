"""Bounded evidence prompts and strict citation validation before display."""

import json
import re
from pathlib import Path

from pydantic import ValidationError

from freezer_rag.config import Settings
from freezer_rag.models import Answer, Chunk, GeneratedAnswer, RagError
from freezer_rag.ollama import Ollama
from freezer_rag.retrieval.hybrid import DURATION_QUERY, DURATION_VALUE

PROMPT_PATH = Path(__file__).with_name("system_prompt.txt")
CITATION = re.compile(r"\[S(\d+)\]")


def excerpt(text: str, question: str, byte_limit: int) -> str:
    """Choose complete relevant source lines where possible; never invent a summary."""
    if len(text.encode("utf-8")) <= byte_limit:
        return text
    stop = {"the", "is", "a", "an", "of", "to", "in", "and", "what", "which", "how", "does"}
    terms = set(re.findall(r"[a-z0-9_/-]+", question.lower())) - stop
    lines = [line for line in re.split(r"\n+|(?<=[.!?])\s+", text) if line.strip()]
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
    settings: Settings, question: str, chunks: list[Chunk], history: list[tuple[str, str]]
) -> tuple[list[dict], dict[str, Chunk]]:
    schema = GeneratedAnswer.model_json_schema()
    system = PROMPT_PATH.read_text(encoding="utf-8")
    budget = settings.llm_num_ctx - settings.llm_num_predict
    messages = [{"role": "system", "content": system}]
    final = {"role": "user", "content": json.dumps({"question": question, "evidence": []})}
    if prompt_cost([*messages, final], schema) > budget:
        raise RagError("Question/instructions exceed the context budget. Shorten the question.")
    # Bound history relative to remaining space, not the total prompt size; otherwise
    # fixed system/schema overhead can accidentally exclude every prior turn.
    base_cost = prompt_cost([*messages, final], schema)
    history_budget = min(800, (budget - base_cost) // 3)
    retained: list[dict] = []
    if settings.max_history_turns:
        for user, assistant in reversed(history[-settings.max_history_turns :]):
            turn = [
                {"role": "user", "content": user[:160]},
                {"role": "assistant", "content": assistant[:160]},
            ]
            cost = prompt_cost([*messages, *turn, *retained, final], schema)
            if cost - base_cost > history_budget:
                break
            retained = turn + retained
    messages.extend(retained)
    evidence = []
    sources = {}
    used_chars = 0
    for index, chunk in enumerate(chunks):
        label = f"S{len(sources) + 1}"
        current = {
            "role": "user",
            "content": json.dumps(
                {"question": question, "evidence": evidence},
                ensure_ascii=False,
            ),
        }
        available = budget - prompt_cost([*messages, current], schema)
        count = len(chunks) - index
        allowance = min((available - 96 * count) // count, settings.max_context_chars - used_chars)
        if allowance < 100:
            continue
        text = excerpt(chunk.text, question, allowance)
        content = {"id": label, "source": chunk.source, "text": text}
        candidate = evidence + [content]
        final = {
            "role": "user",
            "content": json.dumps(
                {"question": question, "evidence": candidate},
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
            {"question": question, "evidence": evidence},
            ensure_ascii=False,
        ),
    }
    return [*messages, final], sources


def validate_answer(content: str, sources: dict[str, Chunk]) -> Answer:
    try:
        generated = GeneratedAnswer.model_validate_json(content)
    except ValidationError as exc:
        raise RagError(
            "Model returned an invalid answer structure; no answer was accepted. Retry."
        ) from exc
    ids = set(generated.citation_ids)
    inline = {f"S{number}" for number in CITATION.findall(generated.answer)}
    if not ids.issubset(sources) or (inline and inline != ids):
        raise RagError(
            "Model returned unknown or inconsistent citations; no answer was accepted. Retry."
        )
    if generated.evidence_sufficient and not ids:
        raise RagError(
            "Model claimed sufficient evidence without citations; no answer was accepted."
        )
    if not generated.evidence_sufficient:
        # Do not display model-written unsupported claims in an abstention.
        return Answer(
            text="The indexed documents do not provide enough evidence to answer that. "
            "Current inventory, locations, availability, and actions require an "
            "authorized live Core API, which this document assistant does not use."
        )
    # Citation IDs are the structured contract. Render valid response-level
    # citations ourselves if a model omits inline markers; never accept invented IDs.
    text = generated.answer
    if not inline:
        text += " " + " ".join(f"[{label}]" for label in sources if label in ids)
    return Answer(
        text=text,
        sources={key: value for key, value in sources.items() if key in ids},
        evidence_sufficient=True,
    )


def generate(
    ollama: Ollama, question: str, chunks: list[Chunk], history: list[tuple[str, str]]
) -> Answer:
    messages, sources = build_messages(ollama.settings, question, chunks, history)
    if not sources:
        return Answer(
            text="No usable evidence fits the prompt. Try a more specific question or "
            "increase the context budget."
        )
    schema = GeneratedAnswer.model_json_schema()
    schema["properties"]["citation_ids"]["items"]["enum"] = list(sources)
    content = ollama.chat(messages, schema)
    return validate_answer(content, sources)
