"""Page provenance and table-row extraction for the two supplied text PDFs."""

import re
import warnings
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

from freezer_rag.models import RagError, SourceUnit

HEADING = re.compile(r"^\d+(?:\.\d+)*\.?\s+[A-Z][^\n]{2,100}$")


def load_pdf(path: Path, source_hash: str) -> list[SourceUnit]:
    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise RagError(f"Encrypted source cannot be ingested: {path.name}")
        title = str((reader.metadata or {}).get("/Title") or path.stem)
        units: list[SourceUnit] = []
        section = ""
        with pdfplumber.open(path) as document:
            for number, (raw_page, page) in enumerate(zip(reader.pages, document.pages), 1):
                raw = raw_page.extract_text() or ""
                # Contents are duplicate locators, not primary evidence. Retain covers.
                if re.search(r"(?im)^\s*(?:table of )?contents\s*$", raw):
                    continue
                if not raw.strip():
                    warnings.warn(
                        f"{path.name}, physical page {number}: no extractable text; "
                        "image/diagram evidence is not indexed",
                        stacklevel=2,
                    )
                    continue
                tables = page.find_tables()
                outside = page.filter(
                    lambda obj: (
                        not any(
                            obj.get("x0", -1) >= table.bbox[0] - 1
                            and obj.get("x1", -1) <= table.bbox[2] + 1
                            and obj.get("top", -1) >= table.bbox[1] - 1
                            and obj.get("bottom", -1) <= table.bbox[3] + 1
                            for table in tables
                        )
                    )
                )
                body = outside.extract_text(x_tolerance=2) if tables else raw
                headings = [
                    line.strip()
                    for line in (body or "").splitlines()
                    if HEADING.match(line.strip())
                ]
                if headings:
                    section = headings[-1]
                base = dict(
                    source=path.name,
                    source_hash=source_hash,
                    title=title,
                    version="1.0",
                    page=number,
                    section=section,
                    locator=f"physical PDF page {number}",
                )
                if body and body.strip():
                    units.append(SourceUnit(**base, text=body.strip()))
                for table_index, table in enumerate(tables, 1):
                    rows = [
                        row
                        for row in table.extract(x_tolerance=2)
                        if any((cell or "").strip() for cell in row)
                    ]
                    if not rows:
                        continue
                    # Borders around individual text lines can split one visual row.
                    # The key column may itself have decorative margin subdivisions.
                    key_column = next(
                        index for index, cell in enumerate(rows[0]) if (cell or "").strip()
                    )
                    logical = [rows[0]]
                    for row in rows[1:]:
                        if (
                            not any((cell or "").strip() for cell in row[: key_column + 1])
                            and len(logical) > 1
                        ):
                            previous = logical[-1]
                            logical[-1] = [
                                " ".join(filter(None, [(old or "").strip(), (new or "").strip()]))
                                for old, new in zip(previous, row, strict=True)
                            ]
                        else:
                            logical.append(row)
                    rows = logical
                    # Decorative cell margins can create empty columns and rows.
                    rows = [row for row in rows if any((cell or "").strip() for cell in row)]
                    if not rows:
                        continue
                    columns = [
                        index
                        for index in range(max(map(len, rows)))
                        if any(index < len(row) and (row[index] or "").strip() for row in rows)
                    ]

                    def row_text(row):
                        values = []
                        for index in columns:
                            cell = (row[index] if index < len(row) else "") or ""
                            cell = " ".join(cell.split())
                            if cell and (not values or cell != values[-1]):
                                values.append(cell)
                        return " | ".join(values)

                    header = row_text(rows[0])
                    # Keep adjacent rows together so state comparisons retrieve the
                    # definitions, rather than isolated enum names or one half of a table.
                    values = [row_text(row) for row in rows[1:]]
                    text = "\n".join([section, header, *values])
                    units.append(
                        SourceUnit(
                            **{
                                **base,
                                "locator": f"physical PDF page {number}, table {table_index}",
                            },
                            text=text,
                        )
                    )
        if not units:
            raise RagError(f"No text could be extracted from {path.name}")
        return units
    except RagError:
        raise
    except Exception as exc:
        raise RagError(f"Cannot extract {path.name}: {exc}") from exc
