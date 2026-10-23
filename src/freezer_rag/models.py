"""Small shared contracts; evidence and model output remain separate."""

from typing import Literal

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

    evidence_sufficient: bool
    source_ids: list[str] = Field(max_length=4)


class RequestPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    intent: Literal["document", "mixed", "out_of_scope", "live", "action", "override"]
    query: str = Field(max_length=2000)
    sources: list[
        Literal["business_requirements.pdf", "Software_Design_Document.pdf", "openapi.yaml"]
    ] = Field(max_length=3)


class Answer(BaseModel):
    text: str
    sources: dict[str, Chunk] = Field(default_factory=dict)
    evidence_sufficient: bool = False
