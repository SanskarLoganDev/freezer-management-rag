from pathlib import Path

import pytest

from freezer_rag.ingestion.chunking import chunk_units
from freezer_rag.ingestion.corpus import SOURCES, load_corpus, source_hashes
from freezer_rag.ingestion.openapi_loader import expand_refs
from freezer_rag.models import RagError, SourceUnit


def test_source_corpus_preserves_requirement_rows_pages_and_api_operations():
    directory = Path(__file__).resolve().parents[1] / "docs"
    units = load_corpus(directory, source_hashes(directory))
    assert {unit.source for unit in units} == set(SOURCES)
    rule = next(unit for unit in units if unit.page == 17 and "RULE-01" in unit.text)
    assert "An item must have a reservation before physical check-in." in rule.text
    assert "ID | Business rule" in rule.text
    lifecycle = next(unit for unit in units if unit.page == 24 and "RESERVED" in unit.text)
    assert "awaiting physical check-in." in lifecycle.text
    expiry = next(unit for unit in units if unit.page == 17 and "RULE-03" in unit.text)
    assert "released capacity becomes available." in expiry.text
    operations = [
        unit for unit in units if unit.source == "openapi.yaml" and "operationId" in unit.locator
    ]
    assert len(operations) == 42
    confirmation = next(
        unit
        for unit in operations
        if unit.section == "POST /api/v1/reservation-proposals/{proposal_id}/confirmation"
    )
    assert "confirmed=true" in confirmation.text
    assert "ProposalConfirmation" in confirmation.text
    assert "Idempotency-Key" in confirmation.text


def test_openapi_refs_preserve_cycles_and_reject_external_or_missing_targets():
    document = {
        "components": {"schemas": {"Node": {"child": {"$ref": "#/components/schemas/Node"}}}}
    }
    result = expand_refs({"$ref": "#/components/schemas/Node"}, document)
    assert result["definition"]["child"]["reference_note"] == "Recursive reference"
    with pytest.raises(RagError, match="External"):
        expand_refs({"$ref": "https://example.invalid/schema"}, document)
    with pytest.raises(RagError, match="Unresolved"):
        expand_refs({"$ref": "#/missing"}, document)


def test_oversized_units_keep_tail_locators_headers_and_stable_ids():
    unit = SourceUnit(
        source="spec.pdf",
        source_hash="hash",
        title="Spec",
        page=4,
        section="Rules",
        locator="physical PDF page 4, table row 1",
        text="Rules\nID | Requirement\nFR-28 " + "complete evidence " * 100 + "UNIQUE_END",
    )
    chunks = chunk_units([unit], 400, 40)
    assert len(chunks) > 1
    assert chunks[-1].text.endswith("UNIQUE_END")
    assert all(chunk.page == 4 and "ID | Requirement" in chunk.text for chunk in chunks)
    assert [chunk.id for chunk in chunks] == [chunk.id for chunk in chunk_units([unit], 400, 40)]
