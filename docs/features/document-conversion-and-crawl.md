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
