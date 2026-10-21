# Freezer Management RAG

A local document question-answering project using Ollama and the supplied freezer-management requirements, architecture, and OpenAPI documents.

**Current status: initial scaffold and implementation plan.** Configuration and an environment checker are implemented. Ingestion, indexing, retrieval, chat, and the HTTP service are planned; they are not yet runnable features.

Read [plan.md](plan.md) for the implementation phases, model/database comparison, hardware settings, evaluation criteria, and source-contract discrepancies.

## Recommended starting stack

- Generation: your installed `qwen3.5:4b` through Ollama.
- Embeddings: `nomic-embed-text:v1.5` through Ollama, initially 768 dimensions.
- Vector storage: Qdrant's local persistent Python client; no database server needed for the prototype.
- Retrieval: dense search plus BM25 for exact requirement IDs, endpoint paths, and state names.
- Runtime: Python 3.11+, direct Ollama REST calls through HTTPX, with a FastAPI interface added after CLI validation.

The initial application answers questions about the documents with citations. Current freezer availability, item locations, and storage operations require a later authenticated Core API integration.

## Windows setup

Run these commands in PowerShell from this project folder. Setup downloads Python packages and the selected embedding model; the scaffold itself does not install or pull anything automatically.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
ollama pull nomic-embed-text:v1.5
.\.venv\Scripts\python.exe scripts/check_environment.py
```

Using the virtual environment's Python directly avoids PowerShell activation-policy changes. If Ollama is not running, open Ollama or run `ollama serve` in a separate terminal, then retry the checker. Keep your existing `qwen3.5:4b` installation.

Runtime-only installation is also available:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

The checker calls only Ollama's model-list endpoint. It does not load models, generate text, pull models, or index documents. Use `--offline` to check Python, configuration, and source files without contacting Ollama. A missing selected embedding model is reported as a setup failure with the corresponding pull command.

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
|   |-- ingestion/                # Planned PDF/OpenAPI loaders and chunking
|   |-- embeddings/               # Planned Ollama embedding adapter
|   |-- vectorstore/              # Planned Qdrant adapter and index manifests
|   |-- retrieval/                # Planned dense/BM25 fusion and context selection
|   |-- generation/               # Planned Ollama chat adapter and citations
|   |-- api/                      # Planned FastAPI application and schemas
|   `-- integrations/             # Future authorized Core API adapter
|-- prompts/                      # Planned versioned RAG prompt templates
|-- scripts/check_environment.py  # Implemented setup check
|-- tests/                        # Planned unit/integration coverage
|-- evaluation/                   # Seed questions and evaluation instructions
|-- data/
|   |-- processed/                # Extracted text and normalized chunks
|   |-- manifests/                # Source hashes and index configuration
|   |-- indexes/                  # Persistent Qdrant and BM25 artifacts
|   `-- evaluation/               # Generated benchmark reports
|-- logs/
|-- .env.example
|-- requirements.txt
|-- requirements-dev.txt
|-- pyproject.toml
`-- plan.md
```

Only the two PDFs and `openapi.yaml` belong in the initial ingestion allowlist. Repository instructions, this README, planning notes, prompts, generated outputs, and `.env` files must not enter the document index.

## Development

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m ruff check src scripts
```

Add meaningful tests alongside each implementation phase; `tests/` currently documents the intended coverage. End-to-end RAG validation requires the embedding model to be pulled and the pipeline to be implemented.
