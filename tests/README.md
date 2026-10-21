# Planned test coverage

Add tests as behavior is implemented:

- Ingestion: PDF page provenance, table/requirement boundaries, oversized operation splitting, safe local YAML references, and malformed-source handling.
- Indexing: stable chunk IDs, repeat-ingestion idempotency, removal of deleted chunks, persistence, and refusal to query incompatible model/dimension manifests.
- Retrieval: exact endpoint and requirement-ID lookup, paraphrases, deduplication, and context budgets.
- Generation: citation ID validation, unsupported claims, insufficient evidence, conflicting source passages, and injected instructions in source text.
- Integration: mocked Ollama responses, timeouts, missing models, and dimension mismatch; opt-in live local smoke checks.
- Future Core adapter: verified caller context, pagination, RESERVED/OCCUPIED distinction, backend failures, and explicit confirmation before any write workflow.

No RAG test suite exists yet. The scaffold has Python compilation, package/source metadata, and evaluation-record checks. Run the environment checker after installing dependencies to validate configuration and local model availability; full inference and retrieval checks belong to the later implementation phases.
