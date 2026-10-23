"""Small preflight checks; do not rewrite valid documentation questions."""

import re
import unicodedata

from freezer_rag.models import Answer

OVERRIDE = re.compile(
    r"\bsystem\s+override\b|"
    r"\b(?:ignore|disregard|bypass|override)\b.{0,60}"
    r"\b(?:instructions?|guardrails?|system\s+prompt|previous\s+rules?)\b|"
    r"\b(?:invent|fake|fabricate|make\s+up)\b.{0,40}\b(?:citations?|sources?|references?)\b|"
    r"\b(?:print|read|reveal|dump|display|show)\b.{0,80}"
    r"(?:\.env\b|system\s+prompt\b|private\s+(?:files?|credentials?))",
    re.I,
)
CODING_TASK = re.compile(
    r"\b(?:append|write|generate|implement|solve|provide|give|need)\b.{0,120}"
    r"\b(?:leetcode|two\s*sum|(?:python|javascript|java|c\+\+)\s+(?:solution|script|program))\b",
    re.I,
)


def blocked_request(question: str) -> Answer | None:
    text = re.sub(
        r"\s+",
        " ",
        "".join(
            char
            for char in unicodedata.normalize("NFKC", question)
            if unicodedata.category(char) != "Cf"
        ),
    )
    if OVERRIDE.search(text):
        return Answer(
            text="I cannot bypass the document-only rules, reveal private configuration, "
            "or invent sources. Please ask a question about the freezer documents."
        )
    if CODING_TASK.search(text):
        return Answer(
            text="Please ask the freezer-document question separately. "
            "This assistant does not solve unrelated programming exercises."
        )
    return None
