# Focused test coverage

The suite covers the meaningful boundaries of the document RAG implementation:

- Ingestion: PDF page provenance, table/requirement boundaries, oversized operation splitting, safe local YAML references, and malformed-source handling.
- Indexing: stable chunk IDs, repeat-ingestion idempotency, replacement/persistence, refusal to query incompatible manifests, and restoration from retained compatible snapshots.
- Retrieval: exact endpoint and requirement-ID lookup, paraphrases, deduplication, and context budgets.
- Generation: citation ID validation, unsupported claims, insufficient evidence, conflicting source passages, and injected instructions in source text.
- Integration: mocked Ollama responses, timeouts, truncated generation, missing models, base-model overrides, and dimension mismatch; separate live local smoke checks.
- Terminal: command handling, recoverable failures, cancellation, EOF, source display, and context reset.
- Optional Streamlit: one focused scenario covers reruns without duplicate generation, recovery, private conversations, clearing, and answer-model changes without reopening the index. Skipped if Streamlit is absent.

Run `.\.venv\Scripts\python.exe -m pytest -q`. The 17 test cases use the supplied documents for extraction checks, temporary local indexes, mocked Ollama responses, and optional Streamlit AppTest. They do not require running models or a model-comparison dataset. An automatic fixture fails if dotenv file loading is attempted. Use explicit `Settings(_env_file=None)` in tests. Live smoke checks use `--no-env` separately. Regression checks cover follow-up queries, bounded history, and retaining stated durations in evidence excerpts. No tests call the real freezer application.
