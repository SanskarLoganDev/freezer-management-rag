# Freezer Management RAG

A local document question-answering project using Ollama and the supplied freezer-management requirements, architecture, and OpenAPI documents.

**Current status: phases 1-5 implemented.** Terminal chat is the primary interface; an optional terminal-style Streamlit view uses the same pipeline. Readiness checks and retained-snapshot recovery are included. The former live-integration phase 6 has been removed from scope.

Read [plan.md](plan.md) for implementation details, hardware settings, functional checks, and source-contract discrepancies. Model choice is user-controlled; no model-comparison workflow is required.

## Recommended starting stack

- Generation: your installed `qwen3.5:4b` through Ollama.
- Embeddings: `nomic-embed-text:v1.5` through Ollama, initially 768 dimensions.
- Vector storage: Qdrant's local persistent Python client; no database server needed for the prototype.
- Retrieval: dense search plus BM25 for exact requirement IDs, endpoint paths, and state names.
- Runtime: Python 3.11+, direct Ollama REST calls through HTTPX, and Python terminal commands.

The application answers questions about the documents with citations. The OpenAPI YAML is embedded as reference material only. This project will never connect to the freezer application, call its APIs, inspect current inventory, or perform storage operations. Only the configured Ollama service is contacted for model requests.

## Windows setup

Run these commands in PowerShell from this project folder. Reuse an existing virtual environment if already installed. Installation downloads Python packages; the application never pulls models automatically.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
ollama pull nomic-embed-text:v1.5
.\.venv\Scripts\python.exe scripts/check_environment.py
```

Using the virtual environment's Python directly avoids PowerShell activation-policy changes. If Ollama is not running, open Ollama or run `ollama serve` in a separate terminal, then retry the checker. Keep your existing `qwen3.5:4b` installation.

Install all runtime libraries from the single requirements file, then register the project:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

The checker calls only Ollama's model-list endpoint. It does not load models, generate text, pull models, or index documents. Use `--offline` to check Python, configuration, and source files without contacting Ollama. Use `--no-env` to skip opening `.env` entirely. A missing selected embedding model is reported as a setup failure with the corresponding pull command.

`requirements.txt` lists only the nine direct runtime libraries, including Streamlit. Pip installs their dependencies automatically. Development tools remain in the `dev` extra in `pyproject.toml`.

You manage your own `.env`; the coding assistant must never inspect or edit it. `.env.example` lists the useful model/tuning overrides. Omitted values use defaults in `src/freezer_rag/config.py`. Dimensions and the collection name are derived from the selected embedding model. If an existing personal configuration explicitly overrides them, remove stale overrides yourself before switching models. An incompatible dimension override fails clearly.

Set `RAG_LLM_MODEL` to any installed Ollama chat model to change the base model for answers. `RAG_EMBEDDING_MODEL` independently selects embeddings. Changing only the answer model does not invalidate or rebuild the document index. Temperature and retrieval top-k settings remain configurable.

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
.\.venv\Scripts\python.exe -m freezer_rag --no-env --model qwen3.5:4b chat
.\.venv\Scripts\python.exe -m freezer_rag --no-env --embedding-model qwen3-embedding:0.6b ingest
.\.venv\Scripts\python.exe -m freezer_rag --no-env --embedding-model qwen3-embedding:0.6b chat
```

Install a selected model yourself if missing. Switching embeddings requires ingestion into a compatible index; each model defaults to its own collection. Changing generation model, temperature, or retrieval counts does not require re-embedding. `ingest` skips an already-current index; `ingest --rebuild` forces replacement. Failed builds preserve the previously active snapshot. Source or chunking changes trigger rebuilding.

The default embedding residency is zero to release memory between requests. During a dedicated ingestion run, `--embedding-keep-alive 1m` can reduce repeated model loading; embeddings are explicitly unloaded when ingestion completes. Do not run separate ingestion/chat processes against the same local store at once.

## Readiness and recovery

```powershell
.\.venv\Scripts\python.exe -m freezer_rag --no-env doctor
.\.venv\Scripts\python.exe -m freezer_rag --no-env snapshots
.\.venv\Scripts\python.exe -m freezer_rag --no-env restore SNAPSHOT_ID_FROM_LIST
```

`doctor` verifies installed chat capability, embedding metadata, source/index compatibility, and stored vector counts without generating an answer. `restore` validates the retained snapshot and atomically activates it; failed validation leaves the active pointer unchanged. Restore requires the matching sources, embedding model/digest, and chunk settings. If those changed, select matching configuration or run `ingest` to build a new compatible index. A generation-model change needs no re-ingestion.

For backup, close terminal chat, ingestion, and Streamlit first. Copy the complete `data/indexes/qdrant` and `data/manifests` directories together, plus the three original allowlisted source files and `requirements.txt`. `data/processed` is an optional inspectable export. Do not copy a running Qdrant directory. To recover on another machine, install the required dependencies/models, put sources and data back at their configured paths, run `snapshots`, select a compatible snapshot with `restore`, then run `doctor`. Back up existing data separately before replacing it. The user's configuration stays under their control; these procedures do not copy or inspect `.env`.

## Optional Streamlit view

Streamlit is included in `requirements.txt`. Start it from the repository root after ingestion:

```powershell
.\.venv\Scripts\python.exe -m freezer_rag ui
```

For an explicit run that skips the private configuration file, use `freezer_rag --no-env ui` after `-m`. Open the printed loopback URL (normally `http://127.0.0.1:8501`). The native Streamlit theme is dark and monospaced, with `you>` / `assistant>` messages, expandable evidence, and clear/reset controls. Apply answer-model, temperature, retrieval, or context changes from the sidebar to start a fresh conversation. These overrides stay in that browser session and never write configuration files. Change embeddings in startup configuration, ingest the compatible index, and restart the UI.

The index and BM25 corpus stay open across reruns, and inference is serialized across browser sessions. Completed answers are saved in session memory and are not regenerated on rerun; each session has independent conversation history. No transcripts are written to disk. Close Streamlit before using terminal chat/ingestion against the same local store. Streamlit is optional and introduces no separate RAG API or application API access.

Current Streamlit brings its own server dependencies, including Uvicorn, transitively. These are not listed as direct project requirements, and this project defines no Uvicorn/FastAPI application.

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
|   |-- cli.py                    # Terminal commands, readiness/recovery, optional ui launcher
|   |-- application.py            # Shared conversation service
|   |-- ui_app.py                 # Optional native Streamlit view
|   |-- ui_runtime.py             # Synchronized shared index; independent sessions
|   |-- api/                      # Unused historical placeholder
|   `-- integrations/             # Unused historical placeholder; no application access
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
|-- .streamlit/config.toml        # Loopback, terminal theme, no usage telemetry
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

The focused suite tests source/table provenance, bounded OpenAPI references, stable chunks, atomic index replacement/restoration, compatibility, hybrid retrieval, embedding validation, model overrides, prompt/citation boundaries, timeouts/cancellation, and terminal recovery. When the UI extra is installed, one Streamlit regression scenario checks reruns, failure recovery, session isolation, reset, and model switching; otherwise that scenario is skipped. Tests explicitly prohibit dotenv reads and use mocked Ollama responses; live model smoke checks are separate. All coding-assistant invocations must use `--no-env` (or explicit `Settings(_env_file=None)`), including local smoke checks.
