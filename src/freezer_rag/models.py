"""Small shared contracts; evidence and model output remain separate."""

from pydantic import BaseModel, ConfigDict, Field


class RagError(Exception):
    """An actionable failure that can be shown without a traceback."""


class SourceUnit(BaseModel):
    source: str
    source_hash: str
    title: str
    version: str = ""
    page: int | None = None
    section: str = ""
    locator: str
    text: str


class Chunk(SourceUnit):
    id: str
    normalized_text: str
    requirement_ids: list[str] = Field(default_factory=list)


class GeneratedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    answer: str = Field(min_length=1)
    citation_ids: list[str]
    evidence_sufficient: bool


class Answer(BaseModel):
    text: str
    sources: dict[str, Chunk] = Field(default_factory=dict)
    evidence_sufficient: bool = False
