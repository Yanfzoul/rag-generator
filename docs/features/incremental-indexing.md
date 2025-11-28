# Feature: Incremental Indexing

## Overview
Reuse existing embeddings and metadata when rebuilding indexes, only re-embedding new or changed files and pruning deletions.

## Goals
- Avoid full re-embeds for large corpora.
- Keep artifacts aligned (meta, embeddings, FAISS, manifest) after each run.
- Provide clear reporting on what changed.

## User stories
- As an operator, I rerun the indexer after updating a repo and expect only new/modified files to be processed.
- As a maintainer, I want deleted files removed from the index without manual cleanup.

## Requirements
- Run `python ingest/build_index_hybrid_fast.py --incremental` to enable manifest-based diffing.
- Previous state is read from `<index_name>.meta.parquet`, `<index_name>.embeddings.npy`, and `<index_name>.manifest.json` under `paths.index_root`.
- Manifest records file size and mtime per path; diffing yields new/changed/deleted sets.
- Unchanged rows are kept; only new/changed files are chunked and embedded; deleted paths are removed before writing outputs.
- Indexer reports counts for new/changed/deleted and kept chunks; writes refreshed FAISS/meta/embeddings/manifest.

## Flow
1. Load prior meta/embeddings/manifest if present and valid.
2. Scan current sources and build a fresh manifest.
3. Diff manifests to identify new/changed/deleted files.
4. Embed only the new/changed files; keep prior rows/vectors for unchanged ones.
5. Merge kept and new rows, drop deleted, and write updated artifacts.

## Acceptance criteria
- Incremental runs skip re-embedding unchanged files and log the kept count.
- Deleted files no longer appear in meta after a run with `--incremental`.
- Manifest is updated to reflect the latest scan and matches meta/embeddings lengths.

## Risks / open questions
- Corrupted or mismatched prior artifacts trigger a full rebuild; operators should monitor logs.
- Moving files between sources may appear as delete+add; acceptable for current scope.

## Non-functional targets and validation
- Reuse ratio: for runs with <5% file changes, at least 90% of prior embeddings reused; report counts.
- Runtime: incremental run should be <25% of full rebuild time when changes <10%.
- Disk headroom: ensure available disk >2x new embeddings size before writing; warn otherwise.

## Error handling and logging
- If prior artifacts are missing or mismatched, log cause and fall back to full rebuild with a warning.
- Manifest/meta length mismatch is fatal with clear guidance to rebuild cleanly.
- Per-file failures during re-embed are logged with path; run exits non-zero if any failures remain.
- Summary logs: counts of new/changed/deleted/kept files and chunks.

## Security/privacy/ops
- Index artifacts inherit source access; ensure index directories have restricted permissions.
- Avoid logging content; log paths and counts only.
- Backup/retention of indexes is operator-controlled; document the location (`paths.index_root`).

## Eval checklist
- Run full build, then modify <5% of files and rerun `--incremental`; confirm reuse ratio and timing improvement.
- Delete a file and confirm it disappears from meta/manifest and FAISS row counts drop accordingly.
- Introduce a corrupt manifest to see fallback to full rebuild with a clear warning.
