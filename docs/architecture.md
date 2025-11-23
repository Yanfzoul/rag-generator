# Architecture: Config-Driven RAG

## System context
The stack runs locally via Python CLIs. Configuration selects sources and models; ingestion outputs indexes to `store/`; runtime scripts answer questions using local retrieval and optional external generation.

## Component overview
- Configuration layer: YAML files (e.g., `config.yaml`, `config.test.yaml`) selected via `RAG_CONFIG`. Define paths, sources, embedding/reranker models, retrieval weights, prompt, and generation defaults.
- Acquisition and normalization: optional `ingest/fetch_sources.py`, `ingest/crawl_web.py`, and `ingest/normalize_text.py` gather and clean web content into text directories.
- Document conversion and OCR: `ingest/convert_docs_to_text.py` converts PDFs/Office/images to text with optional OCR and writes outputs under configured `_txt` directories.
- Index builder: `ingest/build_index_hybrid_fast.py` scans sources, chunks code/prose, embeds via SentenceTransformers (producer/consumer pipeline), and writes FAISS + parquet + `embeddings.npy` + `manifest.json`. Supports incremental diffing and size caps.
- Symbol index (optional): `rag/build_symbols.py` scans headers to power identifier boosts in retrieval.
- Retrieval service: `rag/retrieve.py` loads artifacts, performs semantic search, optional BM25 lexical search, identifier boosts, weighted fusion, optional CrossEncoder rerank, and returns top-k rows.
- Chat and generation: `rag/chat.py` builds prompts with citations and token budgeting; supports sessions/history, attachments, and `reuse_context`. `rag/chat_openai.py` swaps generation to an OpenAI-compatible API while keeping the same retrieval path.
- Storage layout: inputs under `data/`, normalized outputs under `*_txt`, indexes under `store/<index_name>.*`, sessions under `sessions/`, saved contexts under `saved_contexts/`.
- Entrypoints and scripts: `launch.sh`, `launch_test.sh`, `launch_chat_openai.sh`, `tools/doctor.py`, `tools/compare_rag.py`.

## Data flows
- Ingestion flow: fetch/crawl (optional) -> normalize/convert to text -> chunk by type (`code_max_lines`, `prose_max_chars`, `min_chunk_chars`, overlaps) -> embed batches -> write FAISS/meta/embeddings -> record manifest for incremental diffing.
- Query flow: question -> embed query -> hybrid retrieval (semantic + optional BM25 + optional identifier) -> optional rerank -> trim to `final_k` -> prompt assembly using template/system message and token budget -> generation via local model or OpenAI-compatible endpoint -> citations printed if requested.
- Attachment flow: user-specified paths are chunked/embedded on demand, merged (or replace when `--attach_only`) with retrieved context before prompting.
- Incremental update: manifest diff determines new/changed/deleted files; kept embeddings are merged with new embeddings, and deletions are pruned before writing fresh artifacts.

## Deployment and operations
- CPU by default; add `--gpu` or `--mps` to indexing/retrieval/chat to select accelerators; scripts fail fast when requested devices are unavailable and may fall back to CPU.
- Dependencies: PyTorch, SentenceTransformers, FAISS, pandas, numpy, rank_bm25 (optional), unstructured/OCR stack for conversion, transformers for generation.
- Configuration hygiene: keep `RAG_CONFIG` set per run; store artifacts under `paths.index_root`; adjust `retrieval.fusion_weights`, `retrieval.initial_k/final_k`, and reranker IDs per corpus.
- Observability: progress bars during embedding, timing/throughput reports after indexing, verbose flags on chat/retrieve for debugging, convert reports for OCR/empty docs.

## Extensibility
- Add sources or adjust chunking via config only; no code changes required.
- Swap embedding/generation/reranker models by editing config or CLI flags.
- Extend scoring by updating fusion weights or symbol prefixes in config.
- Support new pipelines by adding entries under `fetch`, `crawl`, `normalize`, or `convert_docs`.

## Constraints and risks
- Large binary datasets rely on external tools (Poppler/Tesseract) for OCR; ensure system dependencies are installed.
- GPU acceleration depends on compatible PyTorch wheels and drivers; fallback is CPU.
- Token limits on generation enforce truncation; prompts must stay within `max_input_tokens`.
- Running attach-heavy queries can be expensive; limits protect memory and time.
