"""Operation-sized OpenAPI evidence with bounded, local-only reference expansion."""

import json
from pathlib import Path

import yaml

from freezer_rag.models import RagError, SourceUnit

METHODS = {"get", "post", "put", "patch", "delete", "head", "options", "trace"}


def expand_refs(
    value, document: dict, seen: tuple = (), depth: int = 0, budget: list[int] | None = None
):
    budget = budget if budget is not None else [2500]
    budget[0] -= 1
    if depth > 8 or budget[0] < 0:
        ref = {"$ref": value["$ref"]} if isinstance(value, dict) and "$ref" in value else {}
        return {**ref, "reference_note": "Expansion limit; see the original OpenAPI contract"}
    if isinstance(value, list):
        return [expand_refs(item, document, seen, depth + 1, budget) for item in value]
    if not isinstance(value, dict):
        return value
    if "$ref" in value:
        ref = value["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/"):
            raise RagError(f"External/invalid OpenAPI reference is not allowed: {ref}")
        if ref in seen:
            return {"$ref": ref, "reference_note": "Recursive reference"}
        target = document
        try:
            for part in ref[2:].split("/"):
                target = target[part.replace("~1", "/").replace("~0", "~")]
        except (KeyError, TypeError) as exc:
            raise RagError(f"Unresolved OpenAPI reference: {ref}") from exc
        expanded = expand_refs(target, document, (*seen, ref), depth + 1, budget)
        siblings = {
            key: expand_refs(item, document, seen, depth + 1, budget)
            for key, item in value.items()
            if key != "$ref"
        }
        return {"$ref": ref, "definition": expanded, **siblings}
    return {
        key: expand_refs(item, document, seen, depth + 1, budget) for key, item in value.items()
    }


def load_openapi(path: Path, source_hash: str) -> list[SourceUnit]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or not isinstance(document.get("paths"), dict):
            raise RagError("OpenAPI must contain an object with paths")
        info = document.get("info", {})
        base = dict(
            source=path.name,
            source_hash=source_hash,
            title=info.get("title", path.stem),
            version=info.get("version", ""),
        )
        units = [
            SourceUnit(
                **base,
                section="API overview",
                locator="info.description",
                text=json.dumps(info, ensure_ascii=False, indent=2),
            )
        ]
        security = document.get("components", {}).get("securitySchemes", {})
        for route, path_item in document["paths"].items():
            if not isinstance(path_item, dict):
                raise RagError(f"Invalid path item: {route}")
            for method, operation in path_item.items():
                if method not in METHODS:
                    continue
                if not isinstance(operation, dict):
                    raise RagError(f"Invalid operation: {method} {route}")
                label = f"{method.upper()} {route}"
                operation_id = operation.get("operationId", "")
                content = {
                    "method": method.upper(),
                    "path": route,
                    "path_parameters": path_item.get("parameters", []),
                    "security": operation.get("security", document.get("security", [])),
                    "security_schemes": security,
                    **operation,
                }
                expanded = expand_refs(content, document)
                units.append(
                    SourceUnit(
                        **base,
                        section=label,
                        locator=f"{label}; operationId {operation_id}",
                        text=f"{label}\n" + json.dumps(expanded, ensure_ascii=False, indent=2),
                    )
                )
        return units
    except RagError:
        raise
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        raise RagError(f"Cannot load OpenAPI {path.name}: {exc}") from exc
