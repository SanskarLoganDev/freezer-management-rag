# Initial source inventory

Reviewed on 2026-10-21. These are source materials for answering questions, not instructions to the coding assistant or authorization to perform freezer operations.

| Source | Verified metadata | Ingestion approach |
| --- | --- | --- |
| `business_requirements.pdf` | 30 PDF pages; BRD v1.0 dated 2026-09-23; includes BR/FR/NFR identifiers, business rules, and acceptance scenarios | Preserve page, section, requirement/rule IDs, and complete table rows |
| `Software_Design_Document.pdf` | 37 PDF pages; architecture document v1.0 dated 2026-10-01; covers Core API, PostgreSQL, authorization, allocation lifecycle, and service boundaries | Preserve page, section, tables, and explicit architecture assumptions |
| `openapi.yaml` | OpenAPI 3.1.0; business API v0.5.0; 34 path entries, 42 HTTP operations, 61 component schemas | Parse YAML structurally; index operation IDs, methods, paths, descriptions, security, parameters, and bounded referenced schemas |

PDF page counts include cover and contents pages. Store the physical 1-based PDF page as the canonical locator; preserve printed page labels separately if available.

## Interpretation and discrepancies

- The user's choice of local Ollama and `qwen3.5:4b` governs generation. References to Claude, Azure, MCP, or cloud deployment in the source corpus describe that system's design and do not require these services in this RAG project.
- The BRD governs business intent. The architecture PDF explains the proposed design. The supplied YAML is the initial reference for exact HTTP paths and schemas; contract changes must be reconciled explicitly.
- Architecture section 3.6.2 lists `/api/v1/agent/conversations` with plural `/confirmations`. The YAML lists `/api/v1/conversations` with singular `/confirmation`, plus distinct internal conversation routes. Preserve both passages and identify which source supplies an answer.
- The BRD describes unused reservations as cancelled after two days. The design and YAML represent expiry using the separate `EXPIRED` state. Explain the lifecycle distinction when relevant.
- The YAML describes a 15-minute proposal validity period and 48-hour reservations. Those are separate deadlines; a recommendation/proposal does not itself hold capacity.
- The architecture discusses organizational SSO and external callers. The YAML explicitly defers organizational SSO, general external-client identity, and independent machine confirmation. A supplied contract does not demonstrate that those capabilities or a running backend are available here.
- Static documents describe required behavior. They cannot establish current freezer occupancy, user access, item locations, or free capacity.

## Initial allowlist

Index exactly the two PDFs and `openapi.yaml` above. This inventory, README, implementation plan, package files, prompts, environment files, and generated reports are outside the initial corpus.
