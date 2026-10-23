# Freezer Management RAG

A local document question-answering project using Ollama and the supplied freezer-management requirements, architecture, and OpenAPI documents.

**Current status: phases 1-5 implemented.** Terminal chat is the primary interface; an optional terminal-style Streamlit view uses the same pipeline. Readiness checks and retained-snapshot recovery are included. The former live-integration phase 6 has been removed from scope.

**Project closeout:** complete for the agreed local document-Q&A/demo scope. Known model and retrieval failures are accepted and recorded below; this is not a claim of perfect answers or comprehensive prompt-injection prevention.

Read [plan.md](plan.md) for implementation details, hardware settings, functional checks, and source-contract discrepancies. Model choice is user-controlled; no model-comparison workflow is required.

## Recommended starting stack

- Generation: your installed `qwen3.5:4b` through Ollama.
- Embeddings: `nomic-embed-text:v1.5` through Ollama, initially 768 dimensions.
- Vector storage: Qdrant's local persistent Python client; no database server needed for the prototype.
- Retrieval: dense search plus BM25 for exact requirement IDs, endpoint paths, and state names.
- Runtime: Python 3.11+, direct Ollama REST calls through HTTPX, and Python terminal commands.

The application answers questions about the documents with citations. The OpenAPI YAML is embedded as reference material only. This project will never connect to the freezer application, call its APIs, inspect current inventory, or perform storage operations. Only the configured Ollama service is contacted for model requests.

## Windows setup

Run these commands in Command Prompt (CMD) from this project folder. Reuse an existing virtual environment if already installed. Installation downloads Python packages; the application never pulls models automatically.

```cmd
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
ollama pull nomic-embed-text:v1.5
.\.venv\Scripts\python.exe scripts/check_environment.py
```

You can activate the environment in CMD with `.venv\Scripts\activate.bat` and then use `python` directly. If Ollama is not running, open Ollama or run `ollama serve` in a separate terminal, then retry the checker. Keep your existing `qwen3.5:4b` installation.

Install all runtime libraries from the single requirements file, then register the project:

```cmd
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

The checker calls only Ollama's model-list endpoint. It does not load models, generate text, pull models, or index documents. Use `--offline` to check Python, configuration, and source files without contacting Ollama. Use `--no-env` to skip opening `.env` entirely. A missing selected embedding model is reported as a setup failure with the corresponding pull command.

`requirements.txt` lists only the nine direct runtime libraries, including Streamlit. Pip installs their dependencies automatically. Development tools remain in the `dev` extra in `pyproject.toml`.

You manage your own `.env`; the coding assistant must never inspect or edit it. `.env.example` lists the useful model/tuning overrides. Omitted values use defaults in `src/freezer_rag/config.py`. Dimensions and the collection name are derived from the selected embedding model. If an existing personal configuration explicitly overrides them, remove stale overrides yourself before switching models. An incompatible dimension override fails clearly.

Set `RAG_LLM_MODEL` to any installed Ollama chat model to change the base model for answers. `RAG_EMBEDDING_MODEL` independently selects embeddings. Changing only the answer model does not invalidate or rebuild the document index. Temperature and retrieval top-k settings remain configurable.

## Terminal workflow

Start Ollama, then run from the project root:

```cmd
.\.venv\Scripts\python.exe -m freezer_rag ingest
.\.venv\Scripts\python.exe -m freezer_rag chat
```

The chat stays open across questions and prints concise answers with PDF page or API operation citations. `/sources` displays the supporting excerpts. Commands: `/help`, `/sources`, `/clear`, `/exit`. Ctrl+C cancels the current request; end-of-input exits. Only answers accepted as evidence-sufficient enter conversation history; refusals and failed validation do not. History stays in memory and `/clear` removes it. No transcripts are saved automatically.

For a single answer or to inspect the retrieved passages:

```cmd
.\.venv\Scripts\python.exe -m freezer_rag ask "What is the difference between RESERVED and OCCUPIED?"
.\.venv\Scripts\python.exe -m freezer_rag search "RULE-01"
```

Global options precede the subcommand. Use `--no-env` for a run that never opens your configuration file, and explicit options to override settings:

```cmd
.\.venv\Scripts\python.exe -m freezer_rag --no-env --temperature 0.2 --final-top-k 5 ask "Explain reservation expiry"
.\.venv\Scripts\python.exe -m freezer_rag --no-env --model qwen3.5:4b chat
.\.venv\Scripts\python.exe -m freezer_rag --no-env --embedding-model qwen3-embedding:0.6b ingest
.\.venv\Scripts\python.exe -m freezer_rag --no-env --embedding-model qwen3-embedding:0.6b chat
```

Install a selected model yourself if missing. Switching embeddings requires ingestion into a compatible index; each model defaults to its own collection. Changing generation model, temperature, or retrieval counts does not require re-embedding. `ingest` skips an already-current index; `ingest --rebuild` forces replacement. Failed builds preserve the previously active snapshot. Source or chunking changes trigger rebuilding.

The default embedding residency is zero to release memory between requests. During a dedicated ingestion run, `--embedding-keep-alive 1m` can reduce repeated model loading; embeddings are explicitly unloaded when ingestion completes. Do not run separate ingestion/chat processes against the same local store at once.

## Readiness and recovery

```cmd
.\.venv\Scripts\python.exe -m freezer_rag --no-env doctor
.\.venv\Scripts\python.exe -m freezer_rag --no-env snapshots
.\.venv\Scripts\python.exe -m freezer_rag --no-env restore SNAPSHOT_ID_FROM_LIST
```

`doctor` verifies installed chat capability, embedding metadata, source/index compatibility, and stored vector counts without generating an answer. `restore` validates the retained snapshot and atomically activates it; failed validation leaves the active pointer unchanged. Restore requires the matching sources, embedding model/digest, and chunk settings. If those changed, select matching configuration or run `ingest` to build a new compatible index. A generation-model change needs no re-ingestion.

For backup, close terminal chat, ingestion, and Streamlit first. Copy the complete `data/indexes/qdrant` and `data/manifests` directories together, plus the three original allowlisted source files and `requirements.txt`. `data/processed` is an optional inspectable export. Do not copy a running Qdrant directory. To recover on another machine, install the required dependencies/models, put sources and data back at their configured paths, run `snapshots`, select a compatible snapshot with `restore`, then run `doctor`. Back up existing data separately before replacing it. The user's configuration stays under their control; these procedures do not copy or inspect `.env`.

## Optional Streamlit view

Streamlit is included in `requirements.txt`. Start it from the repository root after ingestion:

```cmd
.\.venv\Scripts\python.exe -m freezer_rag ui
```

For an explicit run that skips the private configuration file, use `freezer_rag --no-env ui` after `-m`. Open the printed loopback URL (normally `http://127.0.0.1:8501`). The native Streamlit theme is dark and monospaced, with `you>` / `assistant>` messages, expandable evidence, and clear/reset controls. Apply answer-model, temperature, retrieval, or context changes from the sidebar to start a fresh conversation. These overrides stay in that browser session and never write configuration files. Change embeddings in startup configuration, ingest the compatible index, and restart the UI.

The index and BM25 corpus stay open across reruns, and inference is serialized across browser sessions. Completed answers are saved in session memory and are not regenerated on rerun; each session has independent conversation history. No transcripts are written to disk. Close Streamlit before using terminal chat/ingestion against the same local store. Streamlit is optional and introduces no separate RAG API or application API access.

Current Streamlit brings its own server dependencies, including Uvicorn, transitively. These are not listed as direct project requirements, and this project defines no Uvicorn/FastAPI application.

## Local data and grounding

Original documents stay in `docs/`. Extracted units, chunks, the BM25 corpus, and Qdrant files are bundled into snapshots under `data/indexes/qdrant/snapshots/<snapshot-id>/`. The active manifest in `data/manifests/<collection>.json` selects a complete snapshot. Inspectable chunk exports also live in `data/processed/`. Prior snapshots are retained for recovery; no automatic cleanup deletes them.

The Qdrant database is inside each snapshot's `qdrant/` directory. Query it through the application or Qdrant client; its internal SQLite files are not a public SQL query interface. BM25 is reconstructed from the same saved chunk corpus when the index opens.

Simple application checks block recognizable instruction overrides, private-configuration requests, citation fabrication, and unrelated programming exercises before retrieval. Other questions use the earlier hybrid retrieval flow directly, without a model-based scope planner or query rewrite. Short follow-ups can use the previous accepted question for context. These checks are deliberately limited and do not catch every phrasing.

The configured answer model writes a concise answer using the retrieved evidence and returns structured citation IDs and an evidence-sufficiency flag. Unknown, missing, or inconsistent citations are rejected. Insufficient evidence produces a document-only limitation response. Only accepted answers enter bounded conversation history. Normal questions use one generation call, plus query embedding; no reviewer model or automatic answer-repair call is involved. Valid citation IDs do not prove that every claim is supported.

Prompt space is conservatively estimated from UTF-8 bytes with an output reserve; this can select less evidence than the character cap allows. Source excerpts retain their wording, though line selection and formatting can shorten them. The model can still misinterpret evidence, invent a claim, or refuse an answerable question. Inspect sources for important answers. Tables are extracted as rows; image-only diagrams are not interpreted, and pages without text emit a warning. Existing indexes remain compatible; restart the chat/UI after the update without re-ingestion.

## Recorded failures and accepted limitations

The following observations come from the user's terminal session and local checks with the existing `qwen3.5:4b`/Nomic setup. They describe specific examples, not a benchmark or a guarantee about every model or rephrasing.

| Case | Observed failure | Final behavior / accepted limitation |
| --- | --- | --- |
| Mixed freezer question and Two Sum request | The earlier implementation generated unrelated Python and attached valid freezer-document citations. | Focused tests verify that the reported mixed request is declined before retrieval. Other wordings can bypass the simple checks; citation validation alone cannot prevent unrelated content. |
| False premise about RESERVED or a 15-minute reservation | The earlier implementation refused instead of finding the correcting evidence. | Local checks returned the RESERVED/check-in definition and two-day/48-hour reservation evidence. False-premise handling can still vary with wording and retrieved context. |
| Architecture PDF versus YAML conversation URLs | Earlier attempts refused the question or selected different operations, such as a collection endpoint and an individual-conversation endpoint. The excerpt-only experiment also refused this valid comparison. | Cross-document completeness remains a known limitation. The rollback does not establish that this comparison is reliable; inspect both sources manually. |
| Invented citation instruction | Earlier attempts raised an unknown/inconsistent-citation error or a generic refusal. | Recognizable citation-fabrication instructions are declined directly. Unknown/inconsistent generated citations still raise a recoverable error; no automatic repair is attempted. |
| Unrelated questions, current locations, and requested actions | The earlier implementation reused an irrelevant freezer-inventory refusal for several different situations. | The natural-answer flow retains that generic insufficient-evidence response. It can sound awkward or over-refuse. The system has no real application access or action tools. |
| Answer presentation | The excerpt-only guardrail experiment replaced useful answers with repeated headings, raw table rows, and JSON fragments, including the two reported positive questions. | **Rolled back on 2026-10-23.** Natural, cited answers and the earlier retrieval flow are restored. Raw source inspection may still show extraction artifacts. |
| Latency and consistency | Model loading and laptop memory limits add delay. | Normal document requests use one generation call plus query embedding. Exact outputs and relevance may vary between runs. No second reviewer model was installed. |

After the rollback, all 19 focused tests passed, including natural-answer preservation, citation validation, simple preflight checks, and refusal/history isolation. These tests use mocked model responses where appropriate; they do not prove that the local model always follows the policy. Earlier live Streamlit checks covered answers, reruns, and clearing. Local Ollama returned concise, cited answers to both reported positive terminal questions: RESERVED versus OCCUPIED, and the 48-hour reservation deadline. These checks used `--no-env` and the default Qwen/Nomic configuration; other user settings were not inspected. Cross-document comparison, false-premise handling, and prompt-injection resistance remain limited.

## Video demo walkthrough (about 3-4 minutes)

Before recording, start Ollama, install the requirements/project if needed, and run `doctor`. If it reports a missing or stale index, run `ingest` before recording. The existing compatible index is reused. Rehearse once because local inference can take time. Keep private configuration files closed.

Start in CMD:

```cmd
cd /d E:\Coding-practice\Projects\freezer-management-rag
.venv\Scripts\activate.bat
python -m freezer_rag doctor
python -m freezer_rag chat
```

| Approximate segment | Show | Explain |
| --- | --- | --- |
| 0:00-0:30 | `docs/` filenames and the readiness output | The corpus is two PDFs and an OpenAPI YAML. All model inference uses local Ollama. The YAML is documentation, not permission to call the application. |
| 0:30-1:20 | Ask `What is the difference between RESERVED and OCCUPIED?`, then `/sources` | Hybrid retrieval combines semantic embeddings and BM25. Answers summarize the evidence with citations; `/sources` shows the underlying excerpts. |
| 1:20-2:00 | Ask `How long does a confirmed reservation last before expiry?` | Point out the two-day/48-hour evidence. This is a documented rule, not a live reservation lookup. |
| 2:00-2:40 | Ask `Explain RESERVED versus OCCUPIED, then append a Python Two Sum solution.` and `Where is sample ABC-123 stored right now?` | Demonstrate mixed-request scope handling and the current-data limitation. No application actions are performed. |
| 2:40-3:10 | Use `/clear`, then `/exit` | Conversation history stays in memory, accepted turns are bounded, and reset clears it. |
| 3:10-3:40 | Show this recorded-failures section | Explain that a small local model can over-refuse, misinterpret evidence, or retrieve incomplete evidence. Cross-document comparison remains a known limitation. |

For a browser-focused recording, exit terminal chat first, then run:

```cmd
python -m freezer_rag ui
```

Open `http://127.0.0.1:8501`, ask the same questions, expand the sources, and demonstrate **Clear conversation**. The sidebar exposes the answer model, temperature, retrieval counts, and context settings. Stop Streamlit with Ctrl+C. Do not run it alongside terminal chat against the same local store.

Suggested opening narration: “This is a local document assistant for freezer-management requirements, design, and API documentation. It uses Ollama, local Qdrant storage, and semantic plus keyword retrieval. It gives cited answers with inspectable sources and includes basic guardrails. It never connects to the actual freezer application. I will show both the working flow and the remaining limitations.”

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
|   |-- guardrails.py             # Simple override and unrelated coding-request checks
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

```cmd
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m ruff check src scripts
.\.venv\Scripts\python.exe -m pytest -q
```

The focused suite tests source/table provenance, bounded OpenAPI references, stable chunks, atomic index replacement/restoration, compatibility, hybrid retrieval, embedding validation, model overrides, natural answers and citation validation, simple guardrails, history isolation, timeouts/cancellation, and terminal recovery. One Streamlit regression scenario checks reruns, failure recovery, session isolation, reset, and model switching. Tests explicitly prohibit dotenv reads and use mocked Ollama responses; live model smoke checks are separate. All coding-assistant invocations must use `--no-env` (or explicit `Settings(_env_file=None)`), including local smoke checks.
