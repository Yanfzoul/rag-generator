# Feature: Attachment-Augmented QA

## Overview
Allow users to attach local files or directories at query time so answers can reference ephemeral context without rebuilding the main index.

## Goals
- Support ad hoc question answering over user-provided files.
- Keep attachment processing bounded with sensible limits.
- Allow attachment-only runs that ignore the stored index.

## User stories
- As a user, I run `python rag/chat.py --q "Summarize these docs" --attach ./data/docs --attach_only --show_sources` to answer using only attached files.
- As a reviewer, I attach a few files and still retrieve from the main index to blend ad hoc and indexed context.

## Requirements
- Flags: `--attach <paths...>`, `--attach_only`, `--attach_limit_files` (default 200), `--attach_limit_chunks` (default 2048).
- Attachments are chunked and embedded on the fly using the same embedding model as the index; limits protect memory/time.
- When `--attach_only` is set, indexed retrieval is skipped; otherwise attachment chunks are merged with retrieved context.
- Applies to both local generation and OpenAI-bridge chat flows.

## Flow
1. Parse attachment paths and enforce file/chunk caps.
2. Read and chunk attachment content, embed using the configured embedder.
3. If not `--attach_only`, run normal retrieval and combine with attachment chunks; otherwise use attachments alone.
4. Build prompt with citations and generate the answer.

## Acceptance criteria
- Attachment caps are enforced; excessive files/chunks are not processed.
- `--attach_only` skips index retrieval entirely while still returning cited answers from attachments.
- Citations identify attachment sources in outputs when `--show_sources` is used.

## Risks / open questions
- Large attachments may still be slow; encourage users to curate inputs.
- Attachment chunking follows general chunking rules; extremely long lines may need preprocessing.
