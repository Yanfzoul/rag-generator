# Feature: Document Conversion and Web Crawl

## Overview
Convert heterogeneous documents and crawled HTML into text that can be indexed, with optional OCR for scans and images.

## Goals
- Support a wide set of document formats without manual preprocessing.
- Provide a repeatable, config-driven pipeline for crawl -> normalize -> convert.
- Make conversion outputs predictable for downstream indexing.

## User stories
- As an operator, I run `ingest/crawl_web.py --from-config` to fetch docs and `ingest/normalize_text.py --from-config` to clean HTML.
- As an operator, I run `ingest/convert_docs_to_text.py --from-config` to convert PDFs/Office/images to text before indexing.
- As a reviewer, I want OCR thresholds configurable so scanned pages are usable without flooding the index with noise.

## Requirements
- `convert_docs` list entries define `src`, `dst`, `glob`, `enable_ocr`, `use_tesseract`, and OCR thresholds; outputs land in `_txt` directories referenced by `sources`.
- Supported formats include PDF, DOC/DOCX, PPT/PPTX, XLS/XLSX, ODT/ODS/ODP, RTF, and common image types (png/jpg/jpeg/gif/bmp/tiff).
- Web pipelines use `crawl` and `normalize` config sections; `ingest/crawl_web.py --from-config` respects seeds/allowlists/depth, and `ingest/normalize_text.py --from-config` converts HTML to text.
- Conversion retains text-only outputs; images are omitted or converted via OCR.
- Conversion and crawl steps can run independently of indexing so outputs can be inspected.

## Flow
1. (Optional) Crawl web sources per config; HTML saved under `data/`.
2. Normalize crawled HTML to text directories.
3. Run `ingest/convert_docs_to_text.py --from-config` to convert documents and images to text with OCR controls.
4. Point `sources` at the resulting `_txt` directories for indexing.

## Acceptance criteria
- Configured `convert_docs` tasks emit text files under the expected `dst` paths; failures are visible in logs.
- Crawled/normalized content is present and readable before indexing.
- OCR settings can be tuned without code changes and take effect on the next conversion run.

## Risks / open questions
- System dependencies (Tesseract, Poppler) must be installed for OCR; document requirements for each platform.
- Conversion quality varies by source; operators should review outputs for critical corpora.

## Non-functional targets and validation
- Throughput: convert 500 PDFs (~5MB each) in <20 minutes on CPU baseline; log per-file timing to spot outliers.
- OCR quality: configurable thresholds must reduce empty/garbage pages; aim for <5% empty-page rate on sampled scanned sets.
- Crawl politeness: enforce robots.txt and rate limits (<1 req/s by default); cap depth and page count per config.
- Outputs: all converted files land in `_txt` under the configured `dst` with UTF-8 text; no binary blobs.

## Error handling and logging
- Per-file errors (missing deps, corrupt files) logged with path and reason; pipeline continues; summary counts emitted.
- Dependency checks (Poppler/Tesseract) fail fast with install hints; exit non-zero if required tools missing.
- Crawl/normalize failures log URL and HTTP status; skipped pages counted.

## Security/privacy/ops
- Respect robots and allow/deny lists; do not crawl authenticated areas unless explicitly configured.
- No cookies/tokens stored; if provided, keep in env vars, not configs.
- Avoid logging document contents; log paths/URLs only. For PII-sensitive corpora, enable stricter globs/excludes.

## Eval checklist
- Run crawl+normalize on a small seed set; verify HTML -> text output and obeyed allow/deny.
- Convert a mixed batch (PDF, DOCX, image scans) with OCR on/off; inspect sample outputs and empty-page rate.
- Induce a missing dependency to confirm clear failure; fix and rerun successfully.
