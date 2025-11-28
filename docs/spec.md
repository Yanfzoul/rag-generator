# Spec: Config-Driven RAG Stack

## Purpose
Provide a config-first retrieval-augmented generation toolkit that ingests repositories, documents, and crawled pages without code edits, builds hybrid retrieval indexes, and exposes retrieval/chat CLIs with attachment support. For educational purpose.

## Goals
- Drive ingestion, retrieval, and generation behavior from YAML config; no code changes to add sources or models.
- Deliver hybrid retrieval that fuses embeddings, BM25, identifier boosts, and optional CrossEncoder reranking.
- Keep indexing incremental to minimize rebuild time and preserve embeddings for unchanged files.
- Allow ad hoc attachments per question with safe limits.
- Support CPU-first operation while enabling GPU/MPS acceleration and an OpenAI-compatible chat bridge.
- Ship runnable scripts and defaults so new users can stand up the pipeline quickly.

## Non-goals
- Building a hosted UI or long-running API service; interaction is through CLI scripts.
- Automating scheduling/cron for ingestion; operators bring their own orchestration.
- Multi-tenant auth, RBAC, or per-user quotas.
- Long-term retention, encryption at rest, or compliance tooling beyond local filesystem usage.

## Personas and scenarios
- Operator / ML engineer: define sources in config, run fetch/crawl/convert, build indexes with `--incremental`, monitor artifacts and timing.
- Developer / analyst: ask questions via `rag/chat.py`, attach local files, view cited answers, compare baseline vs RAG with `tools/compare_rag.py`.
- Reviewer / QA: verify ingestion coverage, check OCR output, and validate citation quality on spot queries.

## Feature set
- Config-driven ingestion and source management (`docs/features/config-driven-ingestion.md`)
- Document conversion and web crawl (`docs/features/document-conversion-and-crawl.md`)
- Incremental indexing (`docs/features/incremental-indexing.md`)
- Hybrid retrieval and reranking (`docs/features/hybrid-retrieval.md`)
- Attachment-augmented QA (`docs/features/attachment-augmented-qa.md`)
- Accelerator-aware execution and generation bridge (`docs/features/acceleration-and-generation.md`)
- Chunking strategy (per-source, meaning-preserving)

## Functional requirements
1. Configuration: read YAML selected by `RAG_CONFIG`, covering paths, sources, chunking, embedding model, retrieval weights, prompt, and generation defaults.
2. Source ingestion: support multiple local roots with include/exclude globs and `treat_as` hints; optional fetch/crawl/normalize/convert flows output normalized text directories consumed by the indexer.
3. Index building: `ingest/build_index_hybrid_fast.py` emits FAISS index, parquet metadata, `embeddings.npy`, and `manifest.json`; honors chunk sizing (global defaults or per-source overrides) and `max_tokens_per_chunk`; CPU default with optional `--gpu` or `--mps`.
4. Incremental runs: when `--incremental` is set, reuse unchanged embeddings using manifest diffing and prune deleted files before writing new artifacts.
5. Retrieval: `rag/retrieve.py` loads FAISS/meta, fuses semantic scores with BM25 (if enabled) and identifier hits, optionally reranks with a CrossEncoder, and returns the top-k rows; printing is available with `--show`.
6. Chat: `rag/chat.py` builds prompts from the configured template, trims to `max_input_tokens`, supports sessions/history, `--initial_k`/`--final_k`, `--use_reranker`, and attachments; `rag/chat_openai.py` keeps local retrieval while delegating generation to an OpenAI-compatible endpoint.
7. Attachments: `--attach` paths are chunked and embedded on the fly with caps `--attach_limit_files` and `--attach_limit_chunks`; `--attach_only` bypasses the stored index; attachment chunking mirrors per-source rules when the path falls under a known source, otherwise uses global defaults.
8. Conversion tasks: `ingest/convert_docs_to_text.py --from-config` runs `convert_docs` entries (with OCR controls); `ingest/crawl_web.py --from-config` and `ingest/normalize_text.py --from-config` handle web-first pipelines.
9. Observability: `tools/doctor.py` performs environment checks; indexing prints throughput/timing; retrieval/chat log device selection and reranker usage; artifacts live under `paths.index_root` and inputs under `paths.data_root`.

## Interface and contract details
- Config schema (YAML): top-level `version`, `paths` (`data_root`, `index_root`, `cache_root`), `sources` list (`id`, `path`/`globs`, `include`/`exclude`, `treat_as`, `chunking` overrides), `convert_docs` (`input`, `output`, `ocr`, `languages`, `include`/`exclude`), `crawl` (`seeds`, `allow`/`deny`, `depth`, `rate_limits`), `indexing` (chunk defaults, `max_tokens_per_chunk` or `tokenizer_cap`), `embedding` (model id, device `cpu/gpu/mps`, batch sizes), `retrieval` (`bm25` toggle, weights, `initial_k`/`final_k`, reranker id, `max_length`), `prompt`/`generation` (template id, model, temperature, max tokens), `attachments` (file/chunk limits). Unknown fields and type mismatches are fatal with field names; per-source overrides cannot omit required base fields.
- CLI flags and IO contracts: ingest/crawl/convert CLIs accept `--from-config` and write normalized text + artifacts under `paths.*`; they exit non-zero on missing inputs, bad OCR dependencies, or validation failures and log the offending item. Index build accepts `--incremental`, `--gpu`, `--mps`, writes FAISS/parquet/embeddings/manifest, and exits non-zero on missing sources or failed writes. Retrieval CLI accepts `--show`, `--bm25`, `--use_reranker`, `--initial_k/--final_k`, prints ranked rows with path + line spans, and exits non-zero on missing index artifacts. Chat CLIs accept session/history options plus `--attach*` flags, stream or print final text with citations, and surface prompt truncation or auth errors clearly.
- Error handling and defaults: config validation runs before execution; missing required keys or incompatible flags are fatal; additive defaults ensure older configs continue to run when new optional fields are added. Model/reranker download errors name the failing id; attachment over-limit errors are reported without aborting the base query; logging notes when BM25/reranker are disabled due to config.
- Versioning/backward compatibility: `version` governs parsing; additive fields preferred; breaking removals require a migration note and a hard error when loading an unsupported version. CLI flags follow the same additive-first policy; deprecated flags emit warnings before removal.
- Config contract table (sample, non-exhaustive):

| Section          | Required keys                       | Optional keys / notes                                  | Default / validation              |
|------------------|-------------------------------------|--------------------------------------------------------|-----------------------------------|
| paths            | `data_root`, `index_root`           | `cache_root`                                           | Paths must exist/writeable        |
| sources[]        | `id`, `path`/`globs`                | `include`/`exclude`, `treat_as`, `chunking` overrides  | `treat_as` in {code, prose, auto} |
| indexing         | `max_tokens_per_chunk`              | `tokenizer_cap`, chunk sizes/overlap                   | Positive ints only                |
| embedding        | `model`                             | `device`, batch sizes                                  | Device in {cpu,gpu,mps}           |
| retrieval        | `initial_k`, `final_k`              | `bm25`, weights, `reranker`, `max_length`              | `final_k` <= `initial_k`          |
| prompt/generation| `template`, `model`                 | `temperature`, `max_tokens`                            | `temperature` 0-1                 |
| attachments      | `attach_limit_files`, `attach_limit_chunks` | `attach_only` flag                                | Limits > 0                        |

- CLI contract table (sample, non-exhaustive):

| CLI                          | Required inputs                      | Key flags                                  | Outputs / exit behavior                       |
|------------------------------|--------------------------------------|--------------------------------------------|-----------------------------------------------|
| ingest/convert/crawl         | `--from-config`                      | `--only` selectors per tool                | Normalized text; non-zero on missing inputs   |
| ingest/build_index_hybrid_fast| existing normalized text            | `--from-config`, `--incremental`, `--gpu/mps` | FAISS/parquet/embeddings/manifest; non-zero on missing sources or writes |
| rag/retrieve.py              | existing index artifacts             | `--show`, `--bm25`, `--use_reranker`, `--initial_k/final_k` | Ranked rows; non-zero on missing artifacts    |
| rag/chat.py                  | existing index artifacts             | `--initial_k/final_k`, `--use_reranker`, `--attach*`, session/history flags | Answer + citations; non-zero on config/artifact errors |
| rag/chat_openai.py           | existing index artifacts, API key    | same as chat + OpenAI endpoint flags       | Local retrieval + remote generation; non-zero on auth/endpoint errors    |

## Architecture and data flow
```
sources (repos/docs/web) --> fetch/convert/normalize --> normalized text under paths.data_root
    --> build_index_hybrid_fast (chunk + embed + BM25) --> FAISS/parquet/embeddings/manifest under paths.index_root
    --> retrieve (rag/retrieve.py) --> ranked chunks
    --> chat (rag/chat.py / rag/chat_openai.py) --> grounded answer + citations
attachments (per question) --> on-the-fly chunk/embed --> merged with retrieve/chat ranking
```
- Storage boundaries: raw/normalized inputs live under `paths.data_root`; index artifacts under `paths.index_root`; temp/cache under `paths.cache_root`. Attachments are read-only and embedded in memory; no persistent storage beyond session.
- Runtime boundaries: ingestion/convert/crawl are batch jobs; index build is CPU-first with optional GPU/MPS for embeddings; retrieval/chat are short-lived CLI runs that load existing artifacts; OpenAI-compatible generation only touches remote LLMs while keeping retrieval local.
- Failure handling: configs validate before work starts; missing inputs/artifacts produce actionable errors pointing to the failing step; incremental builds keep prior artifacts and prune deleted files; if reranker/model download fails, retrieval/chat continue without it and log the downgrade; attachment limit violations return a clear message without crashing the base query.
- Concurrency: per-source ingest/convert steps may run in parallel when orchestrated; index embedding batches are parallelized per device settings; retrieval/chat operate a single request at a time but reuse loaded models within a process. Concurrency uses bounded workers to avoid CPU/memory spikes.
- Resource constraints: CPU-default everywhere; GPU/MPS guarded by flags and fall back when unavailable; embedding batches sized to stay within memory; chunking/token caps enforce prompt size limits; crawling respects rate limits; OCR enabled only when configured.

## Security, privacy, and ops posture
- PII handling: no deliberate PII processing; configs can exclude paths/patterns; converters skip binaries by default. Recommend running on controlled hosts; operators are responsible for source hygiene.
- Auth and remote calls: OpenAI-compatible generation requires explicit API key/env; keys are read from env/CLI only, not stored; failures are surfaced without retrying with other keys. Retrieval stays local; only generation touches remote endpoints.
- Logging and redaction: logs avoid printing chunk contents or attachment payloads; errors include file paths and IDs only. Prompt construction avoids echoing attachment text in logs; verbose debug gated by flags.
- Backup/retention: artifacts live under `paths.index_root` and are not auto-backed up; operators choose backup/retention based on data sensitivity. Attachments are processed in-memory per query and not persisted.
- Operations: `tools/doctor.py` reports environment readiness; failures exit non-zero with actionable messages. CPU-first defaults prevent surprise GPU usage; crawling obeys rate limits to avoid abuse.

## Delivery plan
- M1: Config validation + doctor gating for ingest/build CLIs; Owner: Eng lead; Target: 2025-11-01.
- M2: Ingest/convert/crawl pipelines stable with sample configs; Owner: Ops; Target: 2025-11-02.
- M3: Hybrid retrieval/chat CLIs with reranker and attachment limits enforced; Owner: Eng; Target: 2025-11-03.
- M4: Eval scenarios + default configs published; Owner: PM/QA; Target: 2025-11-06.

## Risks and mitigations
- Remote model dependency outage blocks generation; mitigate with local default model fallback and cached downloads; Owner: Eng.
- Crawling can violate robots or rate limits; mitigate with allow/deny lists and enforced rate limits; Owner: Ops.
- Large corpora can exceed disk/RAM during indexing; mitigate with incremental builds, batch sizing, and disk headroom checks; Owner: Eng.
- PII leakage via logs or artifacts; mitigate with default redaction, config excludes, and log level controls; Owner: Ops.
- Config drift across environments; mitigate with sample configs, schema validation, and `version` gating; Owner: PM/Eng.

## Assumptions
- Operators run on single-tenant, controlled hosts with trusted data.
- Network access for model downloads is available during setup; subsequent runs can operate offline.
- Users provide valid YAML configs and any required API keys for remote generation.
- Attachments fit in memory for on-the-fly embedding; no long-running server process is assumed.

## Decision log
- Default corpora/configs to ship; Owner: PM; Due: 2024-09-20; Status: pending.
- Orchestration approach (cron/CI vs manual runs); Owner: Ops; Due: 2024-09-20; Status: pending.
- Evaluation harness beyond `tools/compare_rag.py`; Owner: QA/Eng; Due: 2024-09-27; Status: pending.

## Chunking strategy (per-source, meaning-preserving)
- Config: global defaults live under `indexing.*` and `max_tokens_per_chunk`. Each `sources:` entry may include `chunking` overrides: `code_max_lines`, `overlap_lines`, `prose_max_chars`, `min_chunk_chars`, and optional `tokenizer_cap` to replace the global token cap. When absent, fall back to defaults.
- Routing: `treat_as: code` uses code strategy; `treat_as: prose` uses prose strategy; `treat_as: auto` keeps the current heuristic (extensions + path hints) to choose code vs prose.
- Code strategy: windowed by `code_max_lines` with `overlap_lines` (defaults 100/10); prefer breaks on blank lines; skip chunks under 5 chars; IDs keep line ranges.
- Prose strategy: sentence-aware splitting when available (NLTK/SpaCy); pack sentences up to `prose_max_chars`, emitting when `min_chunk_chars` is reached or at EOF; fallback to simple char windows if sentence splitting fails.
- Token cap: after chunking, enforce `max_tokens_per_chunk` or `tokenizer_cap`; trim and log when clipping occurs to keep chunks within model limits.
- Deduplication: per-file pass drops identical chunk texts (whitespace-normalized) to reduce overlap-induced redundancy.
- Metadata: continue emitting `source_type`, `repo`, `path`, `start_line`, `end_line`, `url`; capture optional section headers/titles for prose when detectable to aid reranking and prompting.
- Observability: report per-source chunk counts and average sizes; warn when trimmed chunks exceed a threshold; surface attachment chunk counts when attachments are processed.

## Non-functional requirements
- Operates on CPU-only environments; GPU/MPS paths fail fast and fall back when unavailable.
- Index builds process large corpora incrementally without exhausting memory (chunked producer/consumer).
- Retrieval latency remains low (seconds or less on moderate indexes) and deterministic given the same inputs.
- Outputs are reproducible per config and input corpus; binary assets are skipped or converted to text.
- Dependencies and model IDs are explicit in requirements and config; no hidden downloads beyond model weights.

## Success metrics
- Index builds complete for configured sources with manifest present and no missing artifacts.
- Retrieval returns cited context with relevant hits in `final_k` for curated spot-check queries.
- Chat answers include citations, stay within token budgets, and avoid ungrounded statements on sampled prompts.
- Attachment-only runs respect limits and do not access stored indexes.

## Acceptance criteria and eval scenarios
- Ingestion/build CLIs: `ingest/convert_docs_to_text.py --from-config`, `ingest/crawl_web.py --from-config`, and `ingest/build_index_hybrid_fast.py --incremental` complete without fatal warnings; artifacts (`manifest.json`, FAISS, parquet, `embeddings.npy`) present; unchanged files reused and deleted files pruned per manifest diff. Doctor check passes on target machine.
- Config contracts: sample configs validate against expected keys (paths, sources, chunking, models, weights, prompt defaults) and reject unknown/ill-typed fields with clear errors; per-source chunking overrides are honored.
- Retrieval CLI: `rag/retrieve.py --show` returns deterministic top-k for a frozen index; `--bm25 false` drops BM25 scoring; `--use_reranker` reorders results when enabled. Outputs include paths and line spans.
- Chat CLIs: `rag/chat.py` respects `max_input_tokens`, emits citations, enforces `--attach_limit_files/chunks`, and `--attach_only` bypasses stored index. `rag/chat_openai.py` keeps local retrieval while delegating generation and surfaces auth failures cleanly.
- Attachment flow: attaching files under a known source mirrors that source's chunking rules; unknown paths fall back to global defaults; over-limit attachments are rejected with a clear message.
- Sample eval queries (expected to return cited answers): "How do attachments bypass the stored index?", "What triggers incremental reuse vs rebuild?", "Which chunking strategy is used for code vs prose?", "How do I disable BM25 and rely on embeddings only?"
### Acceptance/eval by persona
- Operator / ML engineer: run end-to-end ingest + index on a sample corpus twice with `--incremental`; second run reuses unchanged embeddings and prunes deletions; doctor passes; logs show per-source chunk counts and timing.
- Developer / analyst: run `rag/retrieve.py --show` on curated queries and see cited line spans; run `rag/chat.py --attach_only` with a local file and get cited answers within token limits; reranker flag visibly reorders results.
- Reviewer / QA: spot-check OCR outputs and chunk metadata for sample files; verify BM25-off behavior (`--bm25 false`) yields embedding-only ranking; confirm attachment over-limit produces a user-facing message without crashing.

