# Freezer Management RAG

A local document question-answering project using Ollama and the supplied freezer-management requirements, architecture, and OpenAPI documents.

**Current status: phases 1-3 implemented.** PDF/OpenAPI ingestion, persistent Qdrant indexing, hybrid retrieval, grounded answers, and interactive terminal chat are available. Streamlit is an optional later interface; there is no RAG HTTP server.

Read [plan.md](plan.md) for implementation details, hardware settings, functional checks, and source-contract discrepancies. Model choice is user-controlled; no model-comparison workflow is required.

## Recommended starting stack

- Generation: your installed `qwen3.5:4b` through Ollama.
- Embeddings: `nomic-embed-text:v1.5` through Ollama, initially 768 dimensions.
- Vector storage: Qdrant's local persistent Python client; no database server needed for the prototype.
- Retrieval: dense search plus BM25 for exact requirement IDs, endpoint paths, and state names.
- Runtime: Python 3.11+, direct Ollama REST calls through HTTPX, and Python terminal commands.

The initial application answers questions about the documents with citations. Current freezer availability, item locations, and storage operations require a later authenticated Core API integration.

## Windows setup

Run these commands in PowerShell from this project folder. Reuse an existing virtual environment if already installed. Installation downloads Python packages; the application never pulls models automatically.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
ollama pull nomic-embed-text:v1.5
.\.venv\Scripts\python.exe scripts/check_environment.py
```

Using the virtual environment's Python directly avoids PowerShell activation-policy changes. If Ollama is not running, open Ollama or run `ollama serve` in a separate terminal, then retry the checker. Keep your existing `qwen3.5:4b` installation.

Runtime-only installation is also available:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

The checker calls only Ollama's model-list endpoint. It does not load models, generate text, pull models, or index documents. Use `--offline` to check Python, configuration, and source files without contacting Ollama. Use `--no-env` to skip opening `.env` entirely. A missing selected embedding model is reported as a setup failure with the corresponding pull command.

You manage your own `.env`; the coding assistant must never inspect or edit it. `.env.example` lists the useful model/tuning overrides. Omitted values use defaults in `src/freezer_rag/config.py`. Dimensions and the collection name are derived from the selected embedding model. If an existing personal configuration explicitly overrides them, remove stale overrides yourself before switching models. An incompatible dimension override fails clearly.

## Terminal workflow

Start Ollama, then run from the project root:

```powershell
.\.venv\Scripts\python.exe -m freezer_rag ingest
.\.venv\Scripts\python.exe -m freezer_rag chat
```

The chat stays open across questions and prints answers with PDF page or API operation citations. Commands: `/help`, `/sources`, `/clear`, `/exit`. Ctrl+C cancels the current request; end-of-input exits. History stays in memory and `/clear` removes it. No transcripts are saved automatically.

For a single answer or to inspect the retrieved passages:

```powershell
.\.venv\Scripts\python.exe -m freezer_rag ask "What is the difference between RESERVED and OCCUPIED?"
.\.venv\Scripts\python.exe -m freezer_rag search "RULE-01"
```

Global options precede the subcommand. Use `--no-env` for a run that never opens your configuration file, and explicit options to override settings:

```powershell
.\.venv\Scripts\python.exe -m freezer_rag --no-env --temperature 0.2 --final-top-k 5 ask "Explain reservation expiry"
.\.venv\Scripts\python.exe -m freezer_rag --no-env --embedding-model qwen3-embedding:0.6b ingest
.\.venv\Scripts\python.exe -m freezer_rag --no-env --embedding-model qwen3-embedding:0.6b chat
```

Install a selected model yourself if missing. Switching embeddings requires ingestion into a compatible index; each model defaults to its own collection. Changing generation model, temperature, or retrieval counts does not require re-embedding. `ingest` skips an already-current index; `ingest --rebuild` forces replacement. Failed builds preserve the previously active snapshot. Source or chunking changes trigger rebuilding.

The default embedding residency is zero to release memory between requests. During a dedicated ingestion run, `--embedding-keep-alive 1m` can reduce repeated model loading; embeddings are explicitly unloaded when ingestion completes. Do not run separate ingestion/chat processes against the same local store at once.

## Local data and grounding

Original documents stay in `docs/`. Extracted units, chunks, the BM25 corpus, and Qdrant files are bundled into snapshots under `data/indexes/qdrant/snapshots/<snapshot-id>/`. The active manifest in `data/manifests/<collection>.json` selects a complete snapshot. Inspectable chunk exports also live in `data/processed/`. Prior snapshots are retained for recovery; no automatic cleanup deletes them.

The Qdrant database is inside each snapshot's `qdrant/` directory. Query it through the application or Qdrant client; its internal SQLite files are not a public SQL query interface. BM25 is reconstructed from the same saved chunk corpus when the index opens.

Answers use bounded excerpts of the retrieved evidence. Prompt space is conservatively estimated from UTF-8 bytes, including instructions, schema, history, and reserved output; this can select less evidence than the character cap allows. Sources in an answer identify the exact excerpts supplied to the model. Citation validation rejects nonexistent/inconsistent IDs, but it cannot prove every model claim is supported; inspect sources for important answers. Tables are extracted as rows; image-only diagrams are not interpreted, and pages without text emit a warning.

## Structure

```text
freezer-management-rag/
|-- docs/                         # Original source corpus and source notes
|   |-- business_requirements.pdf
|   |-- Software_Design_Document.pdf
|   |-- openapi.yaml
|   `-- source_inventory.md
|-- src/freezer_rag/
|   |-- config.py                 # Implemented environment configuration
|   |-- ingestion/                # PDF/OpenAPI loaders and chunking
|   |-- embeddings/               # Ollama embedding adapter
|   |-- vectorstore/              # Qdrant snapshots and index manifests
|   |-- retrieval/                # Dense/BM25 rank fusion
|   |-- generation/               # Bounded prompts and citation validation
|   |-- cli.py                    # ingest, search, ask, and chat
|   |-- application.py            # Shared conversation service
|   |-- api/                      # Unused historical placeholder
|   `-- integrations/             # Future authorized Core API adapter
|-- prompts/                      # Prompt documentation
|-- scripts/check_environment.py  # Implemented setup check
|-- tests/                        # Focused tests; no live models required
|-- evaluation/                   # Seed questions and evaluation instructions
|-- data/
|   |-- processed/                # Extracted text and normalized chunks
|   |-- manifests/                # Source hashes and index configuration
|   |-- indexes/                  # Persistent Qdrant and BM25 artifacts
|   `-- evaluation/               # Reserved generated-output directory
|-- logs/
|-- .env.example
|-- requirements.txt
|-- pyproject.toml
`-- plan.md
```

Only the two PDFs and `openapi.yaml` belong in the initial ingestion allowlist. Repository instructions, this README, planning notes, prompts, generated outputs, and `.env` files must not enter the document index.

## Development

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m ruff check src scripts
.\.venv\Scripts\python.exe -m pytest -q
```

The focused suite tests source/table provenance, bounded OpenAPI references, stable chunks, atomic index replacement/persistence, compatibility checks, exact-ID hybrid retrieval, embedding formatting/validation, prompt budgets, citation rejection, and terminal recovery. Tests explicitly prohibit dotenv reads and use mocked Ollama responses; live model smoke checks are separate. All coding-assistant invocations must use `--no-env` (or explicit `Settings(_env_file=None)`), including local smoke checks.
