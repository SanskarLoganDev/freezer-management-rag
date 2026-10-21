"""Deterministic chunks that retain source locators and table/operation context."""

import re
import unicodedata
from uuid import NAMESPACE_URL, uuid5

from freezer_rag.models import Chunk, SourceUnit

PIPELINE_VERSION = "4"
REQUIREMENT = re.compile(r"\b(?:RULE|BR|FR|NFR|C)-\d+\b", re.IGNORECASE)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Prefer paragraph/line boundaries; never discard the tail of a source unit."""
    parts = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = max(
                text.rfind("\n\n", start + size // 2, end),
                text.rfind("\n", start + size // 2, end),
                text.rfind(" ", start + size // 2, end),
            )
            if boundary > start:
                end = boundary
        part = text[start:end].strip()
        if part:
            parts.append(part)
        if end == len(text):
            break
        start = max(start + 1, end - min(overlap, (end - start) // 3))
    return parts


def chunk_units(units: list[SourceUnit], size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for unit in units:
        context = f"{unit.section}\n{unit.locator}\n"
        if "table " in unit.locator:
            # Repeat column headers when an oversized table row must be split.
            context += "\n".join(unit.text.splitlines()[:2]) + "\n"
        context = context[: min(400, size // 3)]
        remaining = max(32, size - len(context))
        for index, text in enumerate(
            split_text(unit.text, remaining, min(overlap, remaining // 3))
        ):
            original = context + text
            normalized = normalize(original)
            identity = f"{PIPELINE_VERSION}:{unit.source}:{unit.locator}:{index}:{normalized}"
            chunks.append(
                Chunk(
                    **{**unit.model_dump(), "text": original},
                    id=str(uuid5(NAMESPACE_URL, identity)),
                    normalized_text=normalized,
                    requirement_ids=sorted(set(REQUIREMENT.findall(original))),
                )
            )
    # Repeated boilerplate in a source may otherwise create duplicate IDs.
    return list({chunk.id: chunk for chunk in chunks}.values())
