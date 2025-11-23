# Project Plan

## Milestones
1. Spec and architecture sign-off (Day 0-1): deliver spec/architecture/project/feature docs; confirm feature set and configs.
2. Data onboarding (Day 1-3): finalize source definitions, run fetch/crawl/convert, verify outputs in `_txt` folders, run doctor.
3. Indexing and retrieval baseline (Day 3-4): run `ingest/build_index_hybrid_fast.py --incremental`, check artifacts, smoke-test `rag/retrieve.py` queries, tune fusion weights and k-values.
4. Chat and attachment polish (Day 4-5): validate `rag/chat.py` and `rag/chat_openai.py` flows, ensure citations and limits, confirm session handling, test `--attach_only`.
5. Quality and release (Day 5-7): curate sample queries, run `tools/compare_rag.py`, document known issues, update README/release notes, prepare automation scripts if needed.

## Work breakdown
- Configuration: maintain `config.yaml`, `config.test.yaml`, optional domain configs; ensure paths and index names map to artifacts.
- Data preparation: manage fetch/crawl/convert tasks, OCR language and confidence thresholds, include/exclude globs.
- Indexing reliability: monitor manifest diffing, batch sizes, file size caps, and device selection flags.
- Retrieval quality: adjust `fusion_weights`, reranker usage, and symbol prefixes; collect relevance judgments.
- Chat and generation: maintain prompt template/system message, token budgets, default generator IDs, OpenAI bridge parameters.
- Tooling and docs: keep launch scripts aligned with configs; ensure doctor/check scripts cover dependencies.

## Testing and validation
- `python tools/doctor.py` before ingestion.
- Dry-run index builds with `--limit` and `--batch` overrides; verify manifest/meta sizes.
- Sample retrievals via `rag/retrieve.py --show` on curated questions.
- Chat smoke tests with and without `--attach` and with the OpenAI bridge; ensure citations show when requested.
- Optional regression via `tools/compare_rag.py` between baseline and RAG answers.

## Risks and mitigations
- OCR/HTML conversions may be noisy: tune `ocr_conf_threshold`/`ocr_image_conf_threshold`, inspect `convert_report.jsonl`, update globs to avoid images when not needed.
- GPU/MPS availability issues: default to CPU, document required wheels/drivers, keep batch sizes small on CPU.
- Large corpora causing slow builds: use `--incremental`, size caps, and `--limit` for smoke tests; prune sources via include/exclude.
- Model/API changes: keep model IDs in config, pin requirements, and support overriding via CLI flags.

## Acceptance criteria
- All requested docs exist (spec, architecture, project, per-feature) and reflect current code/config behavior.
- Index builds succeed on provided configs and can rerun incrementally without manual cleanup.
- Retrieval/chat return cited answers for sample queries, including attach-only scenarios.
- Launch scripts and instructions remain accurate for CPU and optional accelerator paths.

## Open issues / follow-ups
- Decide default evaluation set and whether to add automated relevance checks.
- Determine hosting for artifacts or cache management for large models.
- Clarify ownership for ongoing config updates and data refresh cadence.
