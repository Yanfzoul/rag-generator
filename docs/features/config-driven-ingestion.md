# Feature: Config-Driven Ingestion and Source Management

## Overview
Add or modify corpora by editing YAML only, keeping ingestion rules (paths, globs, chunking hints) in configuration instead of code.

## Goals
- Let operators onboard new sources without touching Python files.
- Keep per-source include/exclude rules and `treat_as` hints explicit in config.
- Respect size caps and skip unsupported binaries automatically.

## User stories
- As an operator, I add a repo path under `sources` with include/exclude globs and run the index builder to make it searchable.
- As a maintainer, I switch between `config.yaml` and `config.test.yaml` via `RAG_CONFIG` to target different corpora without editing code.
- As an analyst, I want non-text assets ignored so the index stays clean.

## Requirements
- `sources` entries define `name`, `type`, `path`, `include`/`exclude`, `treat_as`, and optional `max_size_mb` to filter large files.
- Chunking parameters (`indexing.code_max_lines`, `indexing.prose_max_chars`, `min_chunk_chars`, `overlap_lines`) live in config and are applied by the indexer.
- Environment variable `RAG_CONFIG` selects the config file for all tools.
- CLI: `python ingest/build_index_hybrid_fast.py --incremental --batch <n> --max_size_mb <m>` reads the active config and honors include/exclude globs.
- Binary/image types and assembly files are skipped by default; only text-like files or converted text outputs are chunked.

## Flow
1. Set `RAG_CONFIG` to the desired config file.
2. Ensure configured source paths exist (local repo, docs, or normalized `_txt` outputs).
3. Run optional fetch/convert steps if the config includes them.
4. Build the index with `ingest/build_index_hybrid_fast.py` (optionally `--incremental`).

## Acceptance criteria
- Adding a new source entry results in chunks and embeddings written to the index artifacts without code changes.
- Include/exclude globs and `max_size_mb` are respected; skipped files are omitted from meta.
- Chunking strategy matches `treat_as` (code vs prose vs auto) using the configured sizes.

## Risks / open questions
- Missing paths are silently skipped; operators should validate source availability before indexing.
- Very large or noisy globs may slow scanning; keep patterns narrow for big repos.

## Non-functional targets and validation
- Scan budget: handle 100k files with defaults in <10 minutes on CPU; warn when over 1k skipped files or >5% filtered by size/glob.
- Memory budget: chunking runs in bounded batches; no single batch should exceed available RAM (guarded by batch size/config).
- Config validation: unknown fields or wrong types are fatal; include/exclude/treat_as must parse cleanly before execution.

## Error handling and logging
- Missing paths or unreadable files: log with path and reason, skip, and continue; emit summary counts.
- Unsupported/binary files: logged once per type with a skip count to avoid log spam.
- Validation errors: fail fast with field name and expected type; exit non-zero.

## Security/privacy/ops
- Encourage excludes for secrets/PII; defaults skip binaries and common secrets globs when provided.
- Runs are local; no network calls. Ensure permissions restrict access to source roots.
- Keep `RAG_CONFIG` under version control with reviews for new sources.

## Eval checklist
- Add a new source with include/exclude and `treat_as`; verify chunks/embeddings created and meta omits excluded files.
- Run with `max_size_mb` set; confirm large files are skipped and logged.
- Toggle between configs via `RAG_CONFIG` and confirm different source sets are applied without code changes.
