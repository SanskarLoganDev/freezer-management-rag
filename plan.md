# Implementation plan: local Freezer Management RAG

Prepared: 2026-10-21. Status: scaffold prepared; the RAG pipeline is not implemented yet.

## 1. Goal and scope

Build a local retrieval-augmented generation (RAG) assistant that answers questions about the supplied Virtual Laboratory Freezer Management System documents, shows verifiable sources, distinguishes conflicting specifications, and says when evidence is insufficient.

The first release is document Q&A. It can explain storage rules, roles, allocation states, architecture, and API contracts. It cannot determine current laboratory inventory, find actually available storage, or perform reservations from static documents. An authenticated Core API adapter is a separate later phase.

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

- Architecture section 3.6.2 uses `/api/v1/agent/conversations` and plural `/confirmations`; the YAML uses `/api/v1/conversations` and singular `/confirmation`. Future HTTP integration follows the supplied YAML unless the actual Core contract is explicitly updated.
- The BRD calls two-day unused reservations cancelled; the architecture and API lifecycle distinguish `EXPIRED` from user cancellation.
- A 15-minute proposal validity period and a 48-hour confirmed reservation are different deadlines. A proposal holds no capacity.
- `RESERVED` is a confirmed destination awaiting check-in. `OCCUPIED` represents physical storage. Current storage paths and reserved destinations must stay distinct.
- The architecture anticipates organizational SSO and external callers; the YAML explicitly defers organizational SSO, general external-client identity, and independent machine confirmation. A document describing an implementation is not evidence of a running backend in this folder.

See `docs/source_inventory.md` for source locators and additional interpretation notes.

## 3. Hardware budget and initial settings

User-provided hardware: Intel Core i7-11370H at 3.30 GHz, 16 GB RAM, NVIDIA RTX 3050 Laptop GPU with 4 GB VRAM, Windows x64. The screenshot shows about 0.30 TB remaining from 1.40 TB total; this is an approximate screenshot value, not a current free-space measurement. Ollama lists `qwen3.5:4b` at 3.3 GB on disk, and that installation was confirmed.

The 4 GB VRAM budget is the main constraint. A model's download size is not its runtime GPU-memory requirement: context/KV cache, compute buffers, display use, and a second model all need memory. Full GPU residency at the chosen context is not guaranteed. CPU or partial GPU execution is an acceptable fallback; measure actual latency before changing models.

Starting configuration in `.env.example`:

| Setting | Starting value | Reason |
| --- | --- | --- |
| Generation model | `qwen3.5:4b` | Reuse the user's installed model |
| Generation context | 4,096 tokens | Conservative starting context for 4 GB VRAM |
| Output budget / temperature | 512 tokens / 0.1 | Bound work and favor consistent grounded answers |
| Embedding model | `nomic-embed-text:v1.5`, 768 dimensions | Small English-document baseline |
| Embedding batch | 4 chunks | Limit transient memory; benchmark 1, 4, and 8 |
| Chunk size / overlap | 1,600 / 200 characters initially | Approximately paragraph-sized; actual tokenizer counts vary |
| Retrieval candidates | Dense 8 + BM25 8; final 4 | Retain exact IDs and paraphrases while limiting prompt size |
| Evidence budget | Up to 8,000 characters | Initial heuristic within a measured token budget, not a token guarantee |
| Model residency | Embeddings `keep_alive=0`; generation `keep_alive=1m` | Release embedding memory before generation |
| Query concurrency | 1 | Avoid multiplied model context and memory consumption |

Perform ingestion separately from interactive chat. Use one serialized embedding call per batch; release the embedding model after the batch call before generation. Unloading every batch trades speed for memory safety; after profiling, allow a short embedding residency during a dedicated ingestion run and unload at its end. Ollama documents the `keep_alive` mechanism and concurrency-related memory use in its [FAQ](https://docs.ollama.com/faq).

If needed, set `OLLAMA_NUM_PARALLEL=1` in the Ollama server's environment and restart Ollama. This is a server setting, separate from the application's `RAG_` variables. Inspect `ollama ps` and `nvidia-smi` during benchmarking. Reduce evidence/output/context budgets or use CPU embedding execution if GPU pressure persists.

All context settings need measured validation, including prompt instructions, evidence, question, bounded chat history, reasoning if emitted, and output. Character limits alone cannot guarantee a 4,096-token prompt fits.

## 4. Embedding model comparison

These are candidates to benchmark, not a measured ranking on this corpus. Download sizes are the currently listed Ollama artifacts, not RAM/VRAM estimates; tags and quantizations may change. Record the installed digest and returned vector dimensions before indexing.

| Candidate | Default dimensions | Input limit / download | Fit and tradeoff |
| --- | --- | --- | --- |
| **`nomic-embed-text:v1.5` (initial choice)** | 768 | Current Ollama tag advertises 2K context; about 274 MB | Small download and practical baseline for English specifications. Requires consistent retrieval prefixes. Benchmark against newer alternatives before final selection. |
| `embeddinggemma:300m` | 768; upstream supports reduced dimensions | 2K context; about 622 MB for the listed default artifact | Multilingual candidate designed for local devices. Useful alternative, with task-specific query/document formatting and a larger default download than Nomic. |
| `qwen3-embedding:0.6b` | 1,024 maximum; upstream supports 32-1,024 | Upstream 32K input; about 639 MB Ollama artifact | Candidate for multilingual and technical retrieval. More parameters and a larger vector size than this baseline; benchmark local latency rather than assuming it is faster or better. No need to share a family with the generation model. |
| `mxbai-embed-large` | 1,024 | 512-token context; about 670 MB | English retrieval comparison. Short input limit requires smaller chunks, which may split requirements or API descriptions; query instructions differ from Nomic. |

Evidence: [Nomic Ollama tag](https://ollama.com/library/nomic-embed-text), [Nomic model card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5), [EmbeddingGemma Ollama tag](https://ollama.com/library/embeddinggemma), [Google model documentation](https://ai.google.dev/gemma/docs/embeddinggemma), [Qwen Ollama tag](https://ollama.com/library/qwen3-embedding:0.6b), [Qwen 0.6B model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [Mixedbread Ollama tag](https://ollama.com/library/mxbai-embed-large), and [Mixedbread model card](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1).

Formatting and compatibility requirements:

1. For Nomic, prepend `search_document: ` to indexed passages and `search_query: ` to questions. Store the unprefixed original text for citations.
2. Implement separate query/document formatting per model. For Qwen use the documented task instruction for queries; for Mixedbread use its retrieval query instruction. Verify whether the selected Ollama template already handles any prefix before adding it, to avoid applying it twice.
3. Use Ollama's current `POST /api/embed` endpoint with batched `input`, explicit `truncate=false`, and `keep_alive`. Reject overflowing inputs and split them; do not silently truncate evidence. See the [embedding API](https://docs.ollama.com/api/embed).
4. Nomic's upstream long-context capability does not imply the current Ollama tag runs with that entire context. Treat the installed model metadata and API overflow behavior as the operational limit.
5. Probe returned dimensions; normalize consistently for cosine similarity and reject unexpected vectors. Do not reduce dimensions without testing the model's supported reduction/normalization procedure.
6. A change in model digest, prefixes, preprocessing, dimensions, or chunking requires a compatible new index. Even equal-length vectors from different models cannot be mixed.

Recommended experiment: start with Nomic; compare EmbeddingGemma and Qwen 0.6B against it using held-out questions and the same initial chunking. Give Mixedbread an additional smaller-chunk run appropriate to its 512-token limit, and label this as a different chunking experiment. Keep the smallest option meeting the retrieval targets. Pull only the initial model now; download alternatives when running comparisons. Generation remains `qwen3.5:4b` throughout.

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

Question answering: user question -> Ollama query embedding + BM25 search -> rank fusion -> deduplicated evidence with source IDs -> bounded prompt -> Ollama `qwen3.5:4b` -> citation validation -> grounded answer or insufficient-evidence response.

Each chunk should contain a deterministic UUIDv5 identifier, source-relative path, source SHA-256, document title/version, physical page or API method/path/operation ID, section, requirement IDs when present, original text, normalized text, and ingestion version. Use stable local IDs suitable for Qdrant rather than arbitrary hash strings as point IDs.

The index manifest records corpus hashes, extraction/chunking versions, model tag and digest, prefix strategy, vector dimensions, distance metric, collection name, and creation time. Check the manifest at query time and fail clearly on incompatibility.

Citations use server-provided source IDs, such as `[S1]`, resolved to source filename plus PDF page/section or HTTP method/path. The model cannot invent file paths, page numbers, or citation IDs. Citation-ID validation verifies references exist; human evaluation must still check whether the cited passage actually supports the claim.

## 7. Implementation phases and completion criteria

### Phase 0 - Foundation (this change)

- [x] Inspect source files and the user's hardware constraints.
- [x] Confirm the installed generation model without changing its installation.
- [x] Create a `src/freezer_rag` package and folders for ingestion, embeddings, vector storage, retrieval, generation, API, and future integrations.
- [x] Add requirements, package metadata, `.env.example`, validated settings, environment checker, data placeholders, source notes, and seed evaluation questions.
- [x] Prepare this implementation plan and document model/database options.

The scaffold does not install packages, pull an embedding model, index documents, start an API, or modify a live freezer system. Setup commands are in README.md. Dependency ranges need resolution in a clean virtual environment and a frozen environment snapshot after integration validation.

### Phase 1 - Setup and document ingestion

- Create a Python 3.11 virtual environment, install dependencies, copy `.env.example` to `.env`, and pull `nomic-embed-text:v1.5`.
- Run the environment checker, verify Ollama generation and embedding calls, probe embedding dimensions, and record installed model digests. Use timeouts and explicit error messages for missing models, server failures, or invalid output.
- Implement `ingestion/pdf_loader.py` using pypdf with physical page numbers. Use pdfplumber selectively for table extraction; validate requirement rows, state tables, and multi-column text against the PDF.
- Implement `ingestion/openapi_loader.py` using `yaml.safe_load`. Build units per method/path/operation, including descriptions, parameters, request/response schema context, security, and operation IDs. Expand only local references with cycle detection and size/depth limits; keep unresolved/cyclic reference names as locators.
- Implement structure-aware chunking: headings and complete requirements first, then paragraph splitting for oversized units. Carry table headers into split rows and the operation locator into split API units. Cap input against the selected embedding model, including prefixes.
- Exclude cover/contents duplication where detected, preserve original evidence, and write extracted units/chunks and source hashes under `data/`.

Done when: all three sources load without silent omissions; inspected PDF tables and API units retain their locators; no prompt/plan/environment content is ingested; malformed files produce clear errors.

### Phase 2 - Embeddings and persistent indexes

- Implement a narrow HTTPX Ollama embedding adapter, with bounded retries for transient failures and no retries for permanent model/validation errors.
- Create the local Qdrant collection using measured dimensions and cosine distance; embed documents in small batches.
- Add the same normalized corpus to BM25, preserving UUID mappings and exact identifiers such as `FR-28`, endpoint paths, and enum values.
- Implement manifest validation, deterministic upserts, removal of stale chunks, and a controlled full rebuild option. Stage a complete replacement index before making it active; keep the prior index usable if ingestion fails.
- Enforce one process per local store and close clients cleanly. Do not run independent CLI ingestion and API workers against the same directory at once.

Done when: repeat ingestion produces no duplicates, changed/deleted sources update the corpus, data survives process restart, dimension/digest mismatch is rejected, and interrupted indexing leaves a usable prior snapshot.

### Phase 3 - Retrieval and model comparison

- Add dense and BM25 retrieval with reciprocal-rank fusion; raw BM25 and cosine scores must not be added directly.
- Retrieve the configured candidate counts, deduplicate by chunk ID and overlapping evidence, and keep final context within the measured token budget.
- Cover paraphrases, exact rule IDs, endpoint lookups, lifecycle distinctions, cross-document answers, conflicting documents, and unanswerable current-state questions.
- Expand `evaluation/questions.jsonl` to at least 30 questions with manually reviewed source relevance. Separate development/tuning questions from a held-out set.
- Compare dense-only, BM25-only, and hybrid results before trying model-based reranking. Benchmark the embedding candidates from section 4 using separate compatible collections.

Done when: a reproducible report records retrieval metrics, source/config hashes, model digests, index size, and cold/warm latency. Keep the lowest-resource candidate that meets the proposed quality targets; report actual results instead of assuming benchmark rankings transfer to this domain.

### Phase 4 - Grounded answers and CLI

- Implement Ollama `POST /api/chat` with explicit context/output settings, bounded history, and timeouts. Verify model-specific reasoning controls rather than assuming another model family's prompt switches work.
- Add versioned templates under `prompts/`: retrieved evidence is untrusted content, answers require supporting sources, and insufficient evidence is an allowed result. Application permissions are enforced in code.
- Return an answer plus validated citation IDs and an evidence-sufficiency indicator. Reject nonexistent citations or malformed structured output; do not silently return a seemingly grounded answer after validation fails.
- Present source conflicts explicitly. Clearly state that current availability and item locations require a live authorized lookup.
- Add the CLI commands for ingestion, retrieval inspection, and asking questions only once implemented; log request timings and source IDs without storing sensitive raw questions by default.

Done when: representative questions produce readable answers with correct source locators; unsupported/live-state questions abstain; injected source instructions do not override the question or execute tools; the pipeline works locally after setup with no cloud dependency.

### Phase 5 - Local HTTP interface and operational checks

- Add FastAPI health/readiness, question, and citation schemas; readiness checks model/index availability without expensive generation on every probe.
- Bind to loopback by default. Keep index rebuilds in a CLI initially, and serialize model access. Run one worker with Qdrant local; move to a Qdrant server before adding multiple workers or concurrent processes.
- Add bounded input sizes, controlled error responses, optional streaming after citation handling is reliable, and resource/timing measurements.
- Freeze resolved dependencies after clean-environment integration checks and document reindexing/backup/recovery procedures.
- Add a UI only after the CLI/API are useful; choose its framework when that phase is requested.

Done when: local HTTP smoke checks pass, missing dependencies/indexes fail clearly, peak memory is measured on the laptop, and the setup/rebuild instructions are repeatable.

### Phase 6 - Optional live Core integration

- Implement a typed Core HTTP adapter from the supplied OpenAPI contract; discover the actual service origin and authentication configuration at integration time.
- Route document explanations to RAG and current-state requests to fresh authorized Core results. Never treat stored document vectors as live inventory.
- Propagate verified caller context outside model-controlled arguments; follow cursors with unchanged filters. Page lengths are not total counts. Apply project access controls before operational information reaches the model.
- Begin with read-only operations. For any later mutation workflow, add explicit proposal confirmation, idempotency, backend revalidation, and authoritative success handling. A model response cannot grant consent or report success before Core commits.
- Keep fit/temperature/access validation, proximity ranking, concurrency, reservation expiry, and audit transactions in Core. Do not implement these as probabilistic vector search.
- Test `RESERVED` versus `OCCUPIED`, 15-minute proposal versus 48-hour reservation deadlines, expiration, timeouts, denied access, stale results, and conflicts. Follow the existing operation's confirmation authority rather than inventing a machine-confirmation mechanism.

Done when: documented authenticated scenarios work against a separately available Core service and document Q&A remains usable when that service is unavailable.

## 8. Proposed evaluation gates

These are starting targets to validate during implementation, not achieved measurements or stakeholder-approved service levels.

| Measure | Proposed gate |
| --- | --- |
| Relevant evidence Recall@5 | At least 90% of held-out answerable questions have a relevant passage in the first five results |
| MRR@5 | Report alongside Recall@5; compare models on the same questions |
| Exact ID/path lookup | All manually selected requirement-ID and endpoint lookup cases retrieve the correct source |
| Citation correctness | All emitted citation IDs/locators resolve; at least 95% of sampled answer claims are supported by their cited passages |
| Unknown/live-state questions | All curated unavailable-current-state cases avoid invented operational answers |
| Document instruction injection | All curated cases keep source text as data and perform no unauthorized actions |
| Reingestion and restart | No duplicate chunk growth; persistent indexes return equivalent evidence after restart |
| Resource use | No out-of-memory errors in a 20-question single-user run; record peak RAM/VRAM and paging |
| Latency | Report ingestion time, cold load, warm retrieval, time to first token, total response, median and p95; set an SLA only after laptop measurements |

Prefer manually reviewed factual and citation checks for this small corpus. Automated similarity scores or the same 4B model grading itself are not sufficient evidence of correctness. Add reranking, OCR, larger contexts, or larger models only when a measured failure justifies the extra work and memory.

## 9. Dependencies and immediate next step

`requirements.txt` contains the intended implementation dependencies, with bounded version ranges. It deliberately uses direct HTTPX calls to Ollama; no LangChain, LlamaIndex, cloud model SDK, duplicate Ollama SDK, or separate PyTorch/CUDA environment is needed for this baseline. pdfplumber is the table-extraction fallback; FastAPI/uvicorn are for phase 5. Development tools are declared in `pyproject.toml` and installed through `requirements-dev.txt`.

Configuration fields are implemented now; their inference, indexing, and retrieval effects begin when the corresponding pipeline adapters are implemented. The environment checker verifies setup and model presence only, and does not prove that chat or embeddings succeed.

Next: complete phase 1, then implement phases 2-4 as a working CLI before adding a UI or live operational tools. Keep Nomic + Qdrant local as the provisional baseline until the held-out evaluation supplies a reason to change it.

## 10. Scaffold verification performed

- Python compilation passed for all 10 source/checker modules.
- TOML package metadata loaded successfully, including 10 runtime requirements and development extras.
- A clean-resolution pip dry run succeeded for all runtime requirements on Python 3.11 / Windows x64; no packages were installed. This checks dependency resolution, not runtime integration.
- All eight evaluation records parsed, IDs are unique, and referenced API operation IDs match the supplied YAML. PDF page counts and original source file sizes were checked.
- The environment checker help command runs. Full configuration/Ollama checks require installing the declared dependencies; generation, embeddings, and RAG have not been exercised.
- No embedding model was pulled, and the supplied three source files were preserved.
