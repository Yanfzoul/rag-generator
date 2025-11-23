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

## Functional requirements
1. Configuration: read YAML selected by `RAG_CONFIG`, covering paths, sources, chunking, embedding model, retrieval weights, prompt, and generation defaults.
2. Source ingestion: support multiple local roots with include/exclude globs and `treat_as` hints; optional fetch/crawl/normalize/convert flows output normalized text directories consumed by the indexer.
3. Index building: `ingest/build_index_hybrid_fast.py` emits FAISS index, parquet metadata, `embeddings.npy`, and `manifest.json`; honors chunk sizing from `indexing` and `max_tokens_per_chunk`; CPU default with optional `--gpu` or `--mps`.
4. Incremental runs: when `--incremental` is set, reuse unchanged embeddings using manifest diffing and prune deleted files before writing new artifacts.
5. Retrieval: `rag/retrieve.py` loads FAISS/meta, fuses semantic scores with BM25 (if enabled) and identifier hits, optionally reranks with a CrossEncoder, and returns the top-k rows; printing is available with `--show`.
6. Chat: `rag/chat.py` builds prompts from the configured template, trims to `max_input_tokens`, supports sessions/history, `--initial_k`/`--final_k`, `--use_reranker`, and attachments; `rag/chat_openai.py` keeps local retrieval while delegating generation to an OpenAI-compatible endpoint.
7. Attachments: `--attach` paths are chunked and embedded on the fly with caps `--attach_limit_files` and `--attach_limit_chunks`; `--attach_only` bypasses the stored index.
8. Conversion tasks: `ingest/convert_docs_to_text.py --from-config` runs `convert_docs` entries (with OCR controls); `ingest/crawl_web.py --from-config` and `ingest/normalize_text.py --from-config` handle web-first pipelines.
9. Observability: `tools/doctor.py` performs environment checks; indexing prints throughput/timing; retrieval/chat log device selection and reranker usage; artifacts live under `paths.index_root` and inputs under `paths.data_root`.

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

## Open questions
- Which corpora and configs should be shipped as defaults beyond the test examples?
- Should ingestion be orchestrated (cron/CI) or kept manual via scripts?
- Do we need an evaluation harness beyond `tools/compare_rag.py` for regression checks?
