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
