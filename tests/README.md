# Focused test coverage

The suite covers the meaningful boundaries of the phases 1-3 implementation:

- Ingestion: PDF page provenance, table/requirement boundaries, oversized operation splitting, safe local YAML references, and malformed-source handling.
- Indexing: stable chunk IDs, repeat-ingestion idempotency, removal of deleted chunks, persistence, and refusal to query incompatible model/dimension manifests.
- Retrieval: exact endpoint and requirement-ID lookup, paraphrases, deduplication, and context budgets.
- Generation: citation ID validation, unsupported claims, insufficient evidence, conflicting source passages, and injected instructions in source text.
- Integration: mocked Ollama responses, timeouts, missing models, and dimension mismatch; opt-in live local smoke checks.
- Terminal: command handling, recoverable request failures, source display, and context reset.

Run `.\.venv\Scripts\python.exe -m pytest -q`. The 12 test cases use the real supplied documents for extraction checks, temporary local indexes for persistence/retrieval checks, and mocked Ollama responses. They do not require running models or a model-comparison dataset. An automatic fixture fails if dotenv file loading is attempted. Use explicit `Settings(_env_file=None)` in tests. Live terminal smoke checks use `--no-env` separately. Regression checks cover follow-up query handling, bounded history, and retaining stated durations in evidence excerpts.
