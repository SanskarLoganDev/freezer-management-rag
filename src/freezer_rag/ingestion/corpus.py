"""The ingestion allowlist is explicit; repository/configuration files are never read."""

import hashlib
from pathlib import Path

from freezer_rag.ingestion.openapi_loader import load_openapi
from freezer_rag.ingestion.pdf_loader import load_pdf
from freezer_rag.models import RagError, SourceUnit

SOURCES = ("business_requirements.pdf", "Software_Design_Document.pdf", "openapi.yaml")


def source_hashes(directory: Path) -> dict[str, str]:
    hashes = {}
    for name in SOURCES:
        path = directory / name
        if not path.is_file():
            raise RagError(f"Missing source: {path}. Restore the three allowlisted documents.")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def load_corpus(directory: Path, hashes: dict[str, str]) -> list[SourceUnit]:
    units = []
    for name in SOURCES:
        loader = load_openapi if name.endswith(".yaml") else load_pdf
        units.extend(loader(directory / name, hashes[name]))
    return units
