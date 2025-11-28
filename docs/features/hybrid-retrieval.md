# Feature: Hybrid Retrieval and Reranking

## Overview
Combine semantic embedding search with optional lexical BM25, identifier boosts, and optional CrossEncoder reranking to return high-quality contexts.

## Goals
- Blend complementary signals (semantic, lexical, identifier) using configurable weights.
- Keep retrieval fast while allowing reranking for precision.
- Expose knobs for corpus-specific tuning.

## User stories
- As an analyst, I query via `rag/retrieve.py --show` and see cited snippets ranked by hybrid scores.
- As an operator, I enable BM25 or reranker via config/flags to improve results for my corpus.
- As a maintainer, I adjust `fusion_weights` or `initial_k`/`final_k` without code changes.

## Requirements
- Semantic search uses FAISS over embeddings created by `ingest/build_index_hybrid_fast.py` and the configured embedding model.
- Lexical BM25 is enabled when `retrieval.bm25.enabled` is true and `rank_bm25` is installed.
- Identifier boosts use symbols from `store/symbols.parquet` when present; symbol prefixes are configurable.
- Fusion weights come from `retrieval.fusion_weights`; `initial_k` and `final_k` are configurable/overridable.
- Optional CrossEncoder reranker controlled by `--use_reranker` and `--reranker_id` (default `BAAI/bge-reranker-base`).
- CLI flags: `--gpu`/`--mps` for embedding device, `--show` to print results.

## Flow
1. Embed query with the configured embedding model (CPU/GPU/MPS based on flags).
2. Retrieve semantic candidates from FAISS (`initial_k`).
3. (Optional) Compute BM25 scores over meta text when enabled.
4. (Optional) Gather identifier hits from symbols for matched paths/lines.
5. Fuse scores using configured weights; take top candidates.
6. (Optional) Rerank top candidates with CrossEncoder; select `final_k` rows and return with metadata.

## Acceptance criteria
- Retrieval works when BM25 or symbols are absent, defaulting to semantic-only fusion.
- Enabling BM25 or reranker changes scoring paths without errors; device selection succeeds or fails fast with clear messages.
- Returned rows include citation-friendly metadata (source type, path, line range when available).

## Risks / open questions
- Rank_bm25 must be installed for lexical fusion; document dependency.
- Reranker increases latency; use only when precision gains are needed.

## Non-functional targets and validation
- Latency budgets (CPU): hybrid without reranker <1.5s p95 on moderate index (~100k chunks); with reranker <3s p95 for `initial_k<=50`.
- Quality targets: hit@k on a curated eval set improves or holds vs semantic-only; set a minimal bar (e.g., hit@5 >= baseline).
- Resource usage: BM25 memory bounded by meta text size; reranker batch sizes sized to fit device memory.

## Error handling and logging
- Missing BM25 dependency disables BM25 with a warning; retrieval continues semantic-only.
- Missing symbols gracefully skip identifier boosts with a warning.
- Reranker download/initialization failures fall back to non-reranked results and log downgrade.
- Logs include weights used, device selection, and whether reranker/BM25/symbols were active.

## Security/privacy/ops
- Retrieval runs local; no external calls unless reranker/embedding fetch models. Cache models locally; avoid embedding text in logs.
- Ensure index/meta paths have proper permissions; no PII is logged.

## Eval checklist
- Run semantic-only vs hybrid vs hybrid+reranker on the eval set; compare hit@k and latency.
- Toggle BM25 off (`--bm25 false`) and verify score path switches without error.
- Remove symbols or reranker to confirm graceful degradation and logged warnings.
