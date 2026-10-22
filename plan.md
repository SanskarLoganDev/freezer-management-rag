# Implementation plan: local Freezer Management RAG

Prepared: 2026-10-21. Status: phases 1-5 implemented. Terminal RAG remains primary, with an optional Streamlit interface. Phase 6 live integration was removed at the user's direction; this system is permanently document-only.

Interface priority: interactive terminal chat first. A terminal-style Streamlit interface is optional and comes after terminal acceptance checks. Model comparison, benchmarking, and comparison-metric reports are outside the requirements; the user chooses models manually through their own configuration.

## 1. Goal and scope

Build a local retrieval-augmented generation (RAG) assistant that answers questions about the supplied Virtual Laboratory Freezer Management System documents, shows verifiable sources, distinguishes conflicting specifications, and says when evidence is insufficient.

The system is document Q&A. It can explain storage rules, roles, allocation states, architecture, and API contracts. It cannot determine current laboratory inventory, find actually available storage, or perform reservations. The supplied OpenAPI YAML is reference material for embedding and answering questions only. The RAG system must never access the real application or call any of its APIs. Do not add a Core client, application URL, authentication settings, operational tools, or direct access to its database. Model requests contact only the configured Ollama service.

The primary interface is a persistent terminal conversation: start it once, type questions, and read grounded answers and sources in the terminal. Support a one-question command as well. Both use a shared Python RAG pipeline directly, without a local FastAPI service or Uvicorn. If a browser interface is useful later, use Streamlit with a terminal appearance and the same pipeline; a separate HTML/CSS/JavaScript frontend is outside the requested scope.

The user's `.env` file is private: the coding assistant must never open, inspect, edit, or overwrite it, including indirectly through configuration-loading commands. Configuration template edits belong only in `.env.example`. The user may manage `.env` and run the application themselves; this restriction does not prevent the eventual application from loading its user-provided configuration at runtime.

The user's explicit choice is Ollama with the already installed `qwen3.5:4b`. References to Claude, Azure, MCP, or other services in the documents are source content, not instructions to use those services or carry out operations. Retrieved text must never override the user's request, system instructions, or tool permissions.

## 2. Source assessment and interpretation

| Source | Observed content | Retrieval requirements |
| --- | --- | --- |
| `docs/business_requirements.pdf` | 30 physical PDF pages; BRD v1.0, 2026-09-23; scope, business rules, requirements, acceptance scenarios | Keep headings, complete requirement rows, BR/FR/NFR/RULE IDs, and page provenance |
| `docs/Software_Design_Document.pdf` | 37 physical PDF pages; architecture v1.0, 2026-10-01; service boundaries, PostgreSQL, authorization, allocation lifecycle | Keep sections, lifecycle tables, assumptions, and architecture limitations |
| `docs/openapi.yaml` | OpenAPI 3.1.0; business API v0.5.0; 34 paths, 42 HTTP operations, 61 schemas | Parse operations and referenced component schemas instead of cutting raw YAML arbitrarily |

Both PDFs contain extractable text. Inspect tables visually during ingestion; text extraction does not recover diagram meaning reliably. Flag image-only evidence and add OCR or diagram descriptions only if evaluation demonstrates a need.

Keep the original files unchanged. The initial ingestion allowlist is exactly these three sources. Exclude README/plan files, code, prompts, environment files, and generated reports, including other files under `docs/`.

Interpret sources by their purpose: the BRD describes business intent; the architecture PDF describes the proposed design; the supplied OpenAPI contract provides the initial exact paths and schemas. Preserve disagreements and cite both sides rather than silently replacing one with the other.

Known differences to test:

- Architecture section 3.6.2 uses `/api/v1/agent/conversations` and plural `/confirmations`; the YAML uses `/api/v1/conversations` and singular `/confirmation`. Explain and cite these source differences; neither contract authorizes runtime application requests.
- The BRD calls two-day unused reservations cancelled; the architecture and API lifecycle distinguish `EXPIRED` from user cancellation.
- A 15-minute proposal validity period and a 48-hour confirmed reservation are different deadlines. A proposal holds no capacity.
- `RESERVED` is a confirmed destination awaiting check-in. `OCCUPIED` represents physical storage. Current storage paths and reserved destinations must stay distinct.
- The architecture anticipates organizational SSO and external callers; the YAML explicitly defers organizational SSO, general external-client identity, and independent machine confirmation. A document describing an implementation is not evidence of a running backend in this folder.

See `docs/source_inventory.md` for source locators and additional interpretation notes.

## 3. Hardware budget and initial settings

User-provided hardware: Intel Core i7-11370H at 3.30 GHz, 16 GB RAM, NVIDIA RTX 3050 Laptop GPU with 4 GB VRAM, Windows x64. The screenshot shows about 0.30 TB remaining from 1.40 TB total; this is an approximate screenshot value, not a current free-space measurement. Ollama lists `qwen3.5:4b` at 3.3 GB on disk, and that installation was confirmed.

The 4 GB VRAM budget is the main constraint. A model's download size is not its runtime GPU-memory requirement: context/KV cache, compute buffers, display use, and a second model all need memory. Full GPU residency at the chosen context is not guaranteed. CPU or partial GPU execution is an acceptable fallback for the user's selected model.

Starting configuration defaults are defined in `config.py`. `.env.example` exposes the answer/base model (`RAG_LLM_MODEL`), embedding model, and useful experiment controls, including temperature, context/output limits, chunking, embedding batch size, retrieval top-k values, and evidence budget. The answer model can be any installed Ollama chat model and changes independently of the embedding index. Dimensions and collection names default automatically from the selected embedding model; optional explicit overrides are validated. Stable defaults such as local paths and model residency need not be repeated in the template. Keep tuning controls active and easy to change; the user alone manages their actual `.env`.

Starting values:

| Setting | Starting value | Reason |
| --- | --- | --- |
| Generation model | `qwen3.5:4b` | Reuse the user's installed model |
| Generation context | 4,096 tokens | Conservative starting context for 4 GB VRAM |
| Output budget / temperature | 512 tokens / 0.1 | Bound work and favor consistent grounded answers |
| Embedding model | `nomic-embed-text:v1.5`, 768 dimensions | Small English-document baseline |
| Embedding batch | 4 chunks | Limit transient memory; the user may tune this if needed |
| Chunk size / overlap | 1,600 / 200 characters initially | Approximately paragraph-sized; actual tokenizer counts vary |
| Retrieval candidates | Dense 8 + BM25 8; final 4 | Retain exact IDs and paraphrases while limiting prompt size |
| Evidence budget | Up to 8,000 characters | Initial heuristic within a measured token budget, not a token guarantee |
| Model residency | Embeddings `keep_alive=0`; generation `keep_alive=1m` | Release embedding memory before generation |
| Query concurrency | 1 | Avoid multiplied model context and memory consumption |

Perform ingestion separately from interactive chat. Use one serialized embedding call per batch; release the embedding model after the batch call before generation. Unloading every batch trades speed for memory safety; after profiling, allow a short embedding residency during a dedicated ingestion run and unload at its end. Ollama documents the `keep_alive` mechanism and concurrency-related memory use in its [FAQ](https://docs.ollama.com/faq).

If needed, set `OLLAMA_NUM_PARALLEL=1` in the Ollama server's environment and restart Ollama. This is a server setting, separate from the application's `RAG_` variables. Inspect `ollama ps` and `nvidia-smi` when diagnosing memory problems. Reduce evidence/output/context budgets or use CPU embedding execution if GPU pressure persists.

All context settings need measured validation, including prompt instructions, evidence, question, bounded chat history, reasoning if emitted, and output. Character limits alone cannot guarantee a 4,096-token prompt fits.

## 4. Configurable embedding model

Build the RAG system with configuration-driven Ollama model selection. The initial embedding preset is `nomic-embed-text:v1.5` (768 dimensions); the user also intends to install `qwen3-embedding:0.6b` (initial full-size preset: 1,024 dimensions) and personally decide which works better. Generation starts with the installed `qwen3.5:4b`. The user changes their own `.env` when trying different settings; the coding assistant must not access that file.

There is no model-comparison phase, automated model selection, required held-out dataset, or model-ranking/metrics report. Supporting a different configured model is ordinary RAG functionality. Verify returned dimensions and model identity for index compatibility, without treating these checks as a comparison exercise.

References: [Nomic model card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) and [Qwen 0.6B model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).

Formatting and compatibility requirements:

1. For Nomic, prepend `search_document: ` to indexed passages and `search_query: ` to questions. Store the unprefixed original text for citations.
2. Implement separate query/document formatting for Nomic and Qwen. For Qwen use its documented task instruction for queries. Verify whether the selected Ollama template already handles formatting before adding it, to avoid applying it twice.
3. Use Ollama's current `POST /api/embed` endpoint with batched `input`, explicit `truncate=false`, and `keep_alive`. Reject overflowing inputs and split them; do not silently truncate evidence. See the [embedding API](https://docs.ollama.com/api/embed).
4. Nomic's upstream long-context capability does not imply the current Ollama tag runs with that entire context. Treat the installed model metadata and API overflow behavior as the operational limit.
5. Probe returned dimensions; normalize consistently for cosine similarity and reject unexpected vectors. Do not reduce dimensions without testing the model's supported reduction/normalization procedure.
6. A change in model digest, prefixes, preprocessing, dimensions, or chunking requires a compatible new index. Even equal-length vectors from different models cannot be mixed.

After the user changes the embedding model, select an existing compatible index or clearly request re-ingestion into a model-compatible collection. Never query an old model's vectors with the new model. Detect incompatible dimension/collection overrides and explain how to correct them. Changing only the generation model, temperature, or retrieval counts does not require re-embedding documents. No comparison tooling or metric collection is needed for these configuration changes.

## 5. Database and vector-store comparison

The document vector index is separate from the freezer application's authoritative business database. PostgreSQL in the design remains responsible for transactions, permissions, items, reservations, and audit records. A RAG vector store does not replace those tables or establish current availability.

| Option | Local setup and capabilities | Limits and project fit |
| --- | --- | --- |
| **Qdrant local persistent client (initial choice)** | `QdrantClient(path=...)`; persistence and metadata payloads without Docker or a running database server | Good prototype for this small corpus. The local implementation targets development/small datasets; it is not a server-scale ANN deployment. One process owns the local directory. Later switch the adapter to a Qdrant server and explicitly migrate/reindex data. |
| Chroma `PersistentClient` | Embedded persistent collections with metadata filters; explicit external Ollama embeddings supported | Straightforward alternative. Configure supplied embeddings rather than allowing an implicit model download. Embedded storage has process/thread constraints; a multi-process API should use the server deployment. |
| PostgreSQL + pgvector | Vectors, SQL filtering, relational joins, exact search and approximate indexes | Strong later choice if an existing PostgreSQL service and database governance favor one platform. Adds server/extension setup; native Windows setup can require build tools or an approved container/service. The RAG service must not gain direct operational database authority. |
| FAISS + SQLite metadata | In-process vector-search library; add SQLite/JSON metadata and explicit index persistence | Useful experimental baseline. More application work for synchronized metadata, updates, deletion, access filtering, and recovery; FAISS alone is a search library rather than a complete document database. |

Primary references: [Qdrant client and local mode](https://github.com/qdrant/qdrant-client), [Chroma clients](https://cookbook.chromadb.dev/core/clients/), [pgvector project](https://github.com/pgvector/pgvector), and [FAISS documentation](https://github.com/facebookresearch/faiss/wiki).

Recommendation: use Qdrant local now; combine dense results with application-side BM25 over the same authorized corpus. Consider pgvector when integrating with a managed PostgreSQL environment. Choose Chroma if its API better suits the team after a small comparison. Avoid running several databases or adding a separate PostgreSQL server for these three documents alone.

Raw float32 vector storage is `chunk_count * dimensions * 4` bytes: 1,000 768-dimensional vectors are about 3.1 MB before payloads, indexes, and runtime overhead. This illustrative calculation is not a measured chunk count or memory forecast. At this corpus size, model execution and source quality are more important choices than database scale.

## 6. Proposed application flow

Offline ingestion: allowlisted source files -> PDF/YAML extraction -> normalized source units -> structure-aware chunks -> Ollama document embeddings -> persistent Qdrant collection + BM25 corpus + versioned manifest.

Question answering: user question -> Ollama query embedding + BM25 search -> rank fusion -> deduplicated evidence with source IDs -> bounded prompt -> configured Ollama answer model -> citation validation -> grounded answer or insufficient-evidence response.

Interface boundaries: terminal command/chat loop -> shared application pipeline -> extraction/index/retrieval/generation adapters. The optional Streamlit interface calls that same pipeline directly. Keep presentation, conversation state, and terminal commands outside retrieval and generation logic. Neither interface needs a separate RAG HTTP server. HTTPX is used only for Ollama; the freezer application's APIs are never called.

Each chunk should contain a deterministic UUIDv5 identifier, source-relative path, source SHA-256, document title/version, physical page or API method/path/operation ID, section, requirement IDs when present, original text, normalized text, and ingestion version. Use stable local IDs suitable for Qdrant rather than arbitrary hash strings as point IDs.

The index manifest records corpus hashes, extraction/chunking versions, model tag and digest, prefix strategy, vector dimensions, distance metric, collection name, and creation time. Check the manifest at query time and fail clearly on incompatibility.

Citations use application-assigned source IDs, such as `[S1]`, resolved to source filename plus PDF page/section or HTTP method/path. The model cannot invent file paths, page numbers, or citation IDs. Citation-ID validation verifies references exist; human evaluation must still check whether the cited passage actually supports the claim.

## 7. Implementation phases and completion criteria

### Phase 0 - Foundation (this change)

- [x] Inspect source files and the user's hardware constraints.
- [x] Confirm the installed generation model without changing its installation.
- [x] Create a `src/freezer_rag` package and folders for ingestion, embeddings, vector storage, retrieval, generation, API, and future integrations.
- [x] Add requirements, package metadata, `.env.example`, validated settings, environment checker, data placeholders, source notes, and seed evaluation questions.
- [x] Prepare this implementation plan and document model/database options.

The original scaffold did not implement the pipeline. Phases 1-3 provide ingestion and terminal entry points without a separate RAG HTTP server or live freezer operations. The existing `api/` and `integrations/` folders remain unused historical placeholders. The user's existing Python 3.13 environment was reused; phase 4 additionally validates a clean Windows x64/Python 3.13 environment. Python 3.11+ remains the declared runtime, but other versions were not integration-tested here.

### Phase 1 - Setup and document ingestion

Implemented: allowlisted corpus extraction, PDF page/table provenance, bounded local OpenAPI references, deterministic chunks, and a `--no-env` environment-checker option. Source PDFs are unchanged.

- Use a supported Python 3.11+ environment and the terminal/core dependency set described in section 9. The user manages `.env` themselves; do not access it or automatically pull models. All three named Ollama models were found already installed.
- Validate configuration and model calls with explicit test configuration that does not load the user's `.env`. The checker now supports `--no-env`, which every coding-assistant invocation must use. Verify the configured generation and embedding models, probe embedding dimensions, and record model digests for index compatibility, using timeouts and explicit errors for missing models, server failures, or invalid output. Exercising both embedding models is not a prerequisite for completing the RAG system.
- Implement `ingestion/pdf_loader.py` using pypdf with physical page numbers. Use pdfplumber selectively for table extraction; validate requirement rows, state tables, and multi-column text against the PDF.
- Implement `ingestion/openapi_loader.py` using `yaml.safe_load`. Build units per method/path/operation, including descriptions, parameters, request/response schema context, security, and operation IDs. Expand only local references with cycle detection and size/depth limits; keep unresolved/cyclic reference names as locators.
- Implement structure-aware chunking: headings and complete requirements first, then paragraph splitting for oversized units. Carry table headers into split rows and the operation locator into split API units. Cap input against the selected embedding model, including prefixes.
- Exclude cover/contents duplication where detected, preserve original evidence, and write extracted units/chunks and source hashes under `data/`.

Done when: all three sources load without silent omissions; inspected PDF tables and API units retain their locators; no prompt/plan/environment content is ingested; malformed files produce clear errors.

### Phase 2 - Embeddings, persistent indexes, and retrieval

Implemented: validated Ollama embeddings, immutable Qdrant snapshots, an atomic active manifest, a saved JSON corpus for BM25, exact-identifier-aware tokenization with common-word filtering, reciprocal-rank fusion, and process ownership checks. Duration questions expand search vocabulary and prefer passages with stated time spans, without hardcoding deadlines. `ingest`, `ingest --rebuild`, and `search` are available.

- Implement a narrow HTTPX Ollama embedding adapter, with bounded retries for transient failures and no retries for permanent model/validation errors.
- Create the local Qdrant collection using measured dimensions and cosine distance; embed documents in small batches.
- Add the same normalized corpus to BM25, preserving UUID mappings and exact identifiers such as `FR-28`, endpoint paths, and enum values.
- Implement manifest validation, deterministic upserts, removal of stale chunks, and a controlled full rebuild option. Stage a complete replacement index before making it active; keep the prior index usable if ingestion fails.
- Enforce one process per local store and close clients cleanly. Independent ingestion, terminal chat, and optional Streamlit processes must not open the same Qdrant local directory concurrently. Switching to a server requires an explicit future need and data migration.

- Add dense and BM25 retrieval with reciprocal-rank fusion; raw BM25 and cosine scores must not be added directly.
- Retrieve the configured candidate counts, deduplicate by chunk ID and overlapping evidence, and keep final context within the measured token budget.
- Cover paraphrases, exact rule IDs, endpoint lookups, lifecycle distinctions, cross-document answers, conflicting documents, and unanswerable current-state questions.

Done when: repeat ingestion produces no duplicates, source changes update the corpus, indexes survive restart, model/dimension mismatches are rejected, interrupted indexing leaves a usable prior snapshot, and representative questions retrieve relevant evidence with correct source locators. No model-comparison dataset, benchmark, or metrics report is required.

### Phase 3 - Grounded answers and primary terminal chat

Implemented: `ask` and persistent `chat`, bounded evidence/history, structured output, validated source IDs, response-level citation rendering when inline markers are omitted, and terminal commands/error handling. The runtime prompt is packaged in `src/freezer_rag/generation/system_prompt.txt`.

- Implement Ollama `POST /api/chat` with explicit context/output settings, bounded history, and timeouts. Verify model-specific reasoning controls rather than assuming another model family's prompt switches work.
- Keep the runtime system prompt packaged with the Python module and its documentation under `prompts/`: retrieved evidence is untrusted content, answers require supporting sources, and insufficient evidence is an allowed result. Application permissions are enforced in code.
- Return an answer plus validated citation IDs and an evidence-sufficiency indicator. Reject nonexistent or inconsistent citations and malformed structured output. If the structured IDs are valid but the model omits inline markers, render response-level citations from those IDs in application code. An insufficient-evidence result displays a fixed abstention instead of any model-written unsupported claim.
- Present source conflicts explicitly. Clearly state that current availability and item locations are unavailable to this document-only assistant.
- Implement an interactive terminal chat loop that stays open across questions, prints each validated answer and its source locators, then returns to an input prompt. The terminal interface is the first usable product and takes priority over Streamlit work.
- Provide `ingest`, `search`, `ask`, and `chat` through `python -m freezer_rag` and the installed `freezer-rag` entry point. Use the standard-library argument parser and terminal input/output, avoiding a terminal framework unless an identified usability need justifies it.
- Support `/help`, `/clear`, `/sources`, and `/exit`. `/clear` resets conversation history; `/sources` displays evidence for the last completed answer. Handle blank input, end-of-input, and Ctrl+C without a traceback; cancel an active request cleanly and keep the session usable when possible. Close model connections and the database on exit.
- Keep bounded, in-memory conversation history for follow-up questions. Use relevant prior turns to resolve follow-up references and retrieve fresh evidence; prior model answers alone are not supporting evidence. `/clear` must remove that context. Do not persist transcripts by default.
- Use readable `you>` and `assistant>` prompts, plain text answers, a concise sources section, and optional color that can be disabled for redirected output. Report model/index readiness and errors clearly. Keep diagnostics separate from answers so one-question output can be redirected or inspected.
- Honor user-configured model/index selection, temperature, retrieval counts, and context/output limits, with optional explicit per-run overrides. Validate overrides, keep embedding/index metadata compatible, and apply the same settings to both interfaces. Do not require a model-comparison mode or metric recording. Avoid storing sensitive raw questions by default.
- Begin with complete, validated responses. Add optional streaming only after deciding how to avoid presenting unvalidated citations as final answers; streaming is not required for the first terminal release.

Done when: a user can start terminal chat once, ask several questions and a grounded follow-up, inspect sources, clear context, and exit cleanly; the one-question command also works. Representative answers have valid supporting source locators, unsupported/live-state questions abstain, source instructions do not execute tools, and missing models/indexes or failed requests produce recoverable errors. The full flow works locally without a browser, Streamlit, FastAPI, Uvicorn, or a cloud dependency.

### Phase 4 - Terminal acceptance and operational checks

Implemented: `doctor`, `snapshots`, validated atomic `restore`, independent answer-model overrides, focused transport/cancellation/recovery checks, clean-environment dependency validation, and backup/recovery instructions. The terminal core was checked before adding the UI dependency.

- Add inexpensive terminal preflight checks for model/index availability, bounded question/history sizes, controlled error handling, and serialized inference.
- Exercise repeated questions, bounded history and follow-ups, `/clear`, `/sources`, exit/cancellation, Ollama timeouts, missing models/indexes, malformed model output, and index/model mismatches. Use explicit test settings without reading the user's `.env`.
- Verify that a configured model change either uses a compatible index or gives a clear re-ingestion/configuration message. Check ordinary single-user operation for memory failures; model comparisons and resource/latency reports are not deliverables.
- Validate dependencies in a clean environment and document reindexing/backup/recovery procedures. Maintain one `requirements.txt` listing direct runtime libraries only, as requested; do not add lock requirements files.
- Update setup and usage instructions around the terminal workflow. Include Streamlit in the single requirements file, while keeping terminal execution independent of UI imports.

Done when: terminal interaction and failure/recovery checks pass, the RAG system works within the laptop's available resources, and setup/rebuild instructions are repeatable. Streamlit work starts only after these terminal acceptance checks pass. The user handles any personal model comparison independently.

### Phase 5 - Optional terminal-style Streamlit interface

Implemented: `freezer-rag ui` launcher, native terminal theme, shared synchronized index, independent session histories/settings, sources, reset, and rerun/failure-recovery checks. Streamlit is included in the single requirements file; use of the browser interface remains optional. The terminal workflow remains primary.

- Keep use of Streamlit optional after terminal acceptance. Install it with the other runtime libraries in `requirements.txt`; terminal execution does not import it. Use Streamlit's native chat input/messages, session state, and supported theme/font options; do not introduce a separately maintained HTML/CSS/JavaScript frontend.
- Present a terminal-like interface with a dark background, monospaced text, readable contrast, `you>` / `assistant>` labels, a chronological transcript, and a compact sources view. Include clear/reset and model/index status controls. Keep configuration controls focused on useful experiment settings.
- Call the shared Python pipeline directly; do not introduce FastAPI or Uvicorn between Streamlit and the pipeline. Preserve identical grounding, citation validation, context limits, model/index compatibility, and errors across terminal and browser views.
- Handle Streamlit reruns without reingesting documents, reopening the local database repeatedly, or regenerating a completed answer. Maintain conversation history per session and an appropriately synchronized shared resource for the local store/model calls. Keep inference serialized; do not open the same store concurrently from a separate terminal process.
- Bind the optional local browser interface to loopback and verify the workflow on Windows. Treat Streamlit's own server as an implementation detail of that optional interface, not as an additional RAG API requirement.

Done when: the Streamlit interface has the requested terminal appearance, handles repeated turns and reruns correctly, uses the same evidence/answers as the terminal pipeline, and remains optional for installation and use. Reference: [Streamlit chat elements and session state](https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps).

### Phase 6 - Removed: live application integration

Removed at the user's explicit direction. The RAG system will never access the real application or call its APIs, including read-only requests. The YAML is indexed solely to answer documentation questions. There is no replacement integration phase, application API client, credentials requirement, or operational-action workflow. Current-state/action requests receive a document-only limitation response.

## 8. Functional acceptance checks

Check that the RAG system behaves correctly. These checks do not require model rankings, benchmark datasets, or comparative metrics.

| Behavior | Acceptance check |
| --- | --- |
| Relevant retrieval | Representative document questions, exact requirement IDs, and API paths retrieve supporting passages |
| Citation correctness | Emitted citation IDs/locators resolve, and reviewed factual answers are supported by their cited passages |
| Unknown/live-state questions | All curated unavailable-current-state cases avoid invented operational answers |
| Document instruction injection | All curated cases keep source text as data and perform no unauthorized actions |
| Reingestion and restart | No duplicate chunk growth; persistent indexes return equivalent evidence after restart |
| Terminal workflow | Multi-turn chat, grounded follow-ups, source inspection, context reset, recoverable failures, and clean cancellation/exit work without UI/API dependencies |
| Optional Streamlit parity | The same pipeline/configuration produces equivalent grounded answers; reruns do not duplicate completed requests or mix session history |
| Model configuration | Model choice follows user configuration; incompatible embeddings/indexes produce clear errors or a re-ingestion instruction |
| Resource use | Ordinary single-user chat completes without out-of-memory errors; failures are handled clearly |

Use the existing seed questions where useful for factual and citation smoke checks; expanding them into a held-out evaluation suite is not required. Add reranking, OCR, or other complexity only when a concrete functional problem justifies it. The user independently decides which configured model they prefer.

## 9. Dependencies and immediate next step

`requirements.txt` is the only requirements file and contains the eight terminal/core dependencies below plus Streamlit for the implemented browser interface. List direct runtime libraries only; pip resolves transitive dependencies automatically. FastAPI and Uvicorn are not direct project requirements because no separate RAG HTTP service is required. `pyproject.toml` reads its runtime dependencies from this file, registers the terminal entry point, and packages the runtime prompt.

| Existing requirement | Purpose in this project | Revised placement |
| --- | --- | --- |
| `httpx>=0.28.1,<1.0` | Call local Ollama embedding/chat HTTP endpoints, with reusable connections and explicit timeouts | Terminal/core runtime |
| `pypdf>=6.0,<7.0` | Extract PDF text page by page while preserving physical page provenance | Ingestion/core runtime |
| `pdfplumber>=0.11.7,<0.12` | Recover and inspect requirement/state tables when ordinary PDF text extraction is insufficient | Initial ingestion dependency; reassess after extraction quality is measured |
| `PyYAML>=6.0.2,<7.0` | Load the supplied OpenAPI YAML structurally with safe loading | Ingestion/core runtime |
| `qdrant-client>=1.15,<2.0` | Persist vectors, source text, and metadata and perform similarity/filter queries through Qdrant local | Terminal/core runtime |
| `rank-bm25>=0.2.2,<0.3` | Complement semantic retrieval with lexical ranking for requirement IDs, paths, and enums; tokenize identifiers deliberately | Terminal/core runtime |
| `pydantic>=2.11,<3.0` | Validate configuration and future chunk/manifest/answer/citation structures; already used by settings | Terminal/core runtime, independent of FastAPI |
| `pydantic-settings>=2.10,<3.0` | Load and validate runtime configuration with defaults and overrides; the user's actual configuration file stays private | Terminal/core runtime |
| `streamlit>=1.55,<2.0` | Provide the implemented terminal-style browser interface | Included in the single runtime requirements file; browser usage is optional |

Use direct HTTPX calls rather than adding a duplicate Ollama SDK, LangChain, LlamaIndex, cloud SDKs, or a separate PyTorch/CUDA environment. Streamlit is included in `requirements.txt` (`streamlit>=1.55,<2.0`; checked version 1.66.1). Its own runtime serves the app; no separate RAG API or Uvicorn application is defined. Current Streamlit depends transitively on Uvicorn and Starlette; pip installs these automatically rather than listing them as direct requirements. The terminal input/command parser uses Python's standard library. Development tools remain separate in the existing `pyproject.toml` dev extra; the absent `requirements-dev.txt` is not required.

Version policy:

- The existing lower bounds are provisional scaffold baselines, not demonstrated minimum supported versions. Do not claim that the pipeline depends on a particular patch release until a used feature or bug fix establishes that need.
- Upper bounds were intended to avoid unreviewed API changes. Most current ranges allow an entire major release series; they are not exact pins. The `<0.12` limit for pdfplumber and `<0.3` limit for rank-bm25 are more conservative minor-series limits. Review them against actual package compatibility rather than preserving them automatically. Pre-1.0 package behavior cannot be inferred from the number alone.
- Keep accepted version ranges in the single `requirements.txt`; do not add exact-version lock files or list transitive dependencies. Other platforms/Python versions require their own validation. Accepted ranges allow installations to resolve different compatible versions; a successful resolution alone does not establish runtime correctness.
- Review dependency upgrades deliberately, including bug/security fixes, and rerun relevant checks. Do not freeze outdated versions indefinitely or loosen bounds without verification.

References: [HTTPX](https://www.python-httpx.org/), [pypdf text extraction](https://pypdf.readthedocs.io/en/stable/user/extract-text.html), [pdfplumber](https://github.com/jsvine/pdfplumber), [PyYAML](https://pyyaml.org/wiki/PyYAMLDocumentation), [Qdrant client](https://github.com/qdrant/qdrant-client), [BM25 implementation](https://github.com/dorianbrown/rank_bm25), [Pydantic validation](https://docs.pydantic.dev/latest/), [Pydantic settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/), [FastAPI server deployment](https://fastapi.tiangolo.com/deployment/manually/), [Python package versioning](https://packaging.python.org/en/latest/discussions/versioning/), and [pip repeatable installs](https://pip.pypa.io/en/stable/topics/repeatable-installs/).

Configuration fields now drive inference, indexing, and retrieval. The environment checker verifies setup and installed model presence only; it still does not itself prove chat or embeddings succeed. The terminal commands and separate live smoke checks exercise the implemented adapters.

Use terminal chat as the primary workflow; launch the optional Streamlit view when useful. Model choice remains under the user's control, with compatible re-ingestion only when embeddings/source/chunk settings change. Phase 6 is removed: no application API access is permitted. No standalone frontend, separate RAG HTTP API, model-comparison phase, or comparison-metrics report is required. Keep Qdrant local until functional or concurrency needs justify a change.

## 10. Scaffold verification performed

These are historical checks against the original scaffold dependency set. They are retained as history; current implementation verification is described below.

- Python compilation passed for all 10 source/checker modules.
- TOML package metadata loaded successfully, including 10 runtime requirements and development extras.
- A clean-resolution pip dry run succeeded for all runtime requirements on Python 3.11 / Windows x64; no packages were installed. This checks dependency resolution, not runtime integration.
- All eight evaluation records parsed, IDs are unique, and referenced API operation IDs match the supplied YAML. PDF page counts and original source file sizes were checked.
- The environment checker help command runs. Full configuration/Ollama checks require installing the declared dependencies; generation, embeddings, and RAG have not been exercised.
- No embedding model was pulled, and the supplied three source files were preserved.

## 11. Phases 1-3 implementation verification

- Reused the existing Python 3.13 virtual environment, installed the declared development tools, and registered the editable terminal entry point. No models were pulled.
- The focused suite has 12 test cases covering the important extraction, reference expansion, chunking, persistence/replacement, compatibility, retrieval, embedding transport, prompt/citation, history, and terminal boundaries. Tests forbid dotenv reads; they use temporary indexes and mocked Ollama responses where appropriate.
- Ruff checks and `pip check` passed. The installed `freezer-rag` help command is available.
- Ingested exactly the three allowlisted sources into a 720-chunk Nomic snapshot. Checked source hashes against the manifest; original source contents are unchanged. Earlier snapshots remain available for recovery.
- Exercised local Nomic embeddings and `qwen3.5:4b` generation with explicit defaults and `--no-env`. Verified cited RESERVED/OCCUPIED and two-day/48-hour reservation-duration answers, live-inventory abstention, terminal help, clear, and exit. Follow-up retrieval/history and duration-excerpt regressions have focused checks. All 12 focused test cases passed.
- The coding assistant did not read or modify the user's `.env`. These checks originally covered phases 1-3; phases 4-5 verification is recorded below. Model comparisons remain the user's own activity, and live Core operations are permanently outside scope.

## 12. Phases 4-5 verification and usage

- `RAG_LLM_MODEL` is exposed in `.env.example` and drives generation independently of embeddings; missing or embedding-only answer models fail clearly. Model changes leave index compatibility unchanged.
- Clean Windows x64/Python 3.13 installation, focused core tests, and model/index readiness were checked before optional UI work. The current setup uses one requirements file for the nine direct runtime libraries, including Streamlit.
- Snapshot recovery tests cover a damaged active pointer, compatible restoration, and refusal of an incompatible snapshot without changing the active pointer. Stop all store users before backing up the complete index/manifests directories and the three original sources; see README for restoration steps.
- Added meaningful timeout, truncated-output, cancellation/EOF, and UI rerun/session-isolation checks rather than tests for individual settings/widgets. Dotenv reads remain prohibited in every test. UI checks are skipped when the optional dependency is absent.
- The native Streamlit view reuses the same RAG service, serializes retrieval/generation with one shared index, keeps session state separate, and exposes answer-model, temperature, top-k, and context controls without writing files. Use `python -m freezer_rag --no-env ui` for validation without loading private configuration.
- No application API client, URL/token settings, authentication adapter, or live operations were implemented. OpenAPI operations remain searchable document evidence only.
- All 17 focused test cases passed in the clean environment, with `pip check` and Ruff checks passing. After synchronizing conversation reset with inference, the six affected conversation/UI checks passed again. The extra dependency lock files were subsequently removed at the user's request; dependencies are maintained only in `requirements.txt`.
- Live checks used only local Ollama with `--no-env`: readiness reported the 720-chunk Nomic index, a terminal answer cited the 48-hour reservation deadline, and Streamlit answered the project-creation method/path with a YAML citation. The live UI preserved completed results across reruns and cleared history correctly. A headless Chrome review verified the fully loaded dark monospaced layout and loopback-only launcher; validation processes were stopped afterward.
- No private `.env` was read, edited, copied, or loaded during validation. The three source documents were preserved. Validation used Python 3.13; other declared Python versions have not received equivalent runtime checks here.
