# Generic, Config‑Driven RAG (Code + Docs)

Quick links
- SETUP (Linux/macOS/Git Bash/GitLab): SETUP.md

This project is a flexible Retrieval‑Augmented Generation (RAG) stack you can point at any sources (repos, websites, PDFs/Office docs, local folders) and use immediately. No code edits are needed: everything is driven by `config.yaml`.

Core features
- Hybrid retrieval: dense embeddings (FAISS) + optional BM25 lexical + optional identifier boosts + optional CrossEncoder reranker
- Config-driven ingestion: define multiple sources with include/exclude rules and per-source chunking hints
- Incremental indexing: re-embeds only changed files; keeps unchanged vectors
- Attachments at question time: add files/directories without rebuilding the index
- Website crawler + HTML to text + Office/PDF to text conversion
- GPU-optional: CPU by default; add `--gpu` to accelerate on CUDA; Tensor-Core friendly
- MPS-optional: CPU by default; add `--mps` to accelerate on Mac Silicon

## Embedding model cheat sheet

| Role / task | Default pick | Why it works well | Lightweight or open alternatives |
| --- | --- | --- | --- |
| Coding repos, code+doc Q&A | `text-embedding-3-large` | Jointly trained on code and prose; strong cross-language recall | `code-search-ada` for legacy parity, `instructor-xl` when you need instruction-tuned vectors |
| Business analyst workflows (requirements, policies, meeting notes) | `text-embedding-3-large` | Handles long-form business text with high precision | `gte-large` for multilingual corpora, `minilm-l12-v2` for very low latency |
| Solution / enterprise architecture knowledge bases | `text-embedding-3-large` | Keeps architecture decision records, specs, and diagrams-as-text in the same space | `e5-base-v2` when you rely on explicit query/document prompts, `bge-large-en` for recall-heavy design docs |
| Support desks / IT ops tickets | `text-embedding-3-small` | Cost-efficient for high-volume ticket similarity while staying accurate | `bge-small-en` for on-prem, `voyage-law-2` when policy or contract nuance matters |
| Company-wide knowledge hubs, FAQs | `text-embedding-3-large` | One model to normalize heterogeneous corpora | `e5-mistral-7b-instruct` if you run self-hosted GPUs, `multi-qa-mpnet-base-dot-v1` for multilingual FAQs |

---

## Install

Pick one of the two ways below.

A) Generic (CPU or GPU if available)
- Create a virtual env and install the generic requirements:
  - `python -m venv .venv`
  - `source .venv/Scripts/activate` (Git Bash) or `.\.venv\Scripts\Activate.ps1` (PowerShell)
  - `python -m pip install --upgrade pip`
  - `python -m pip install -r requirements.txt`

B) Windows + RTX 2080 (CUDA 11.8)
- Use the curated file (and Python 3.11 recommended for CUDA wheels):
  - `python -m venv .venv`
  - `source .venv/Scripts/activate`
  - `python -m pip install --upgrade pip`
  - `python -m pip install -r requirements_rtx2080.txt`
- Or run the setup script (PowerShell):
  - `powershell -ExecutionPolicy Bypass -File scripts/setup_gpu_win_cu118.ps1`

Optional packages
- BM25 lexical retriever (recommended): already included (rank_bm25)
- Silence Windows symlink warnings from HF Hub:
  - Git Bash: `export HF_HUB_DISABLE_SYMLINKS_WARNING=1`
  - PowerShell: `$env:HF_HUB_DISABLE_SYMLINKS_WARNING='1'`

---

## Quick Start

1) Use the test config and run a preflight check
- Git Bash: `export RAG_CONFIG=config.test.yaml`
- PowerShell: `$env:RAG_CONFIG='config.test.yaml'`
- `python tools/doctor.py`

2) Build the index
- CPU: `python ingest/build_index_hybrid_fast.py --batch 8`
- GPU: `python ingest/build_index_hybrid_fast.py --gpu --batch 64`
- Apple Silicon (MPS): `python ingest/build_index_hybrid_fast.py --mps --batch 32`

3) Retrieve results
- CPU: `python rag/retrieve.py --q "How to use add and mul functions?" --show`
- GPU: `python rag/retrieve.py --gpu --q "How to use add and mul functions?" --show`
- MPS: `python rag/retrieve.py --mps --q "How to use add and mul functions?" --show`

4) Chat with context
- CPU: `python rag/chat.py --q "Show a snippet calling add(2,3) and mul(2,3)" --final_k 3 --show_sources`
- GPU: `python rag/chat.py --gpu --q "Show a snippet calling add(2,3) and mul(2,3)" --final_k 3 --show_sources`
- MPS: `python rag/chat.py --mps --q "Show a snippet calling add(2,3) and mul(2,3)" --final_k 3 --show_sources`

5) Attach documents without rebuilding
- CPU: `python rag/chat.py --q "Summarize these docs" --attach ./data/docs --attach_only --attach_limit_files 100 --attach_limit_chunks 1000 --final_k 5 --show_sources`
- GPU: add `--gpu` to the command
- MPS: add `--mps`

6) Compare baseline vs RAG
- CPU: `python tools/compare_rag.py --q "How to init SGDK and draw a sprite?" --rag_final_k 3 --show_sources`
- GPU: add `--gpu`
- MPS: add `--mps`

A helper `launch.sh` shows the same steps and an end-to-end “Docs pipeline” example.

### V1 highlights

- **Broad document coverage** – PDFs (native + OCR fallback), DOC/DOCX (Mammoth), PPT/PPTX, Excel/ODS, HTML/Markdown, and standalone scans/images all flow through `convert_docs_to_text.py`.
- **Tunables per source** – `convert_docs` entries accept `ocr_lang`, `ocr_conf_threshold`, and `ocr_image_conf_threshold` so you can denoise scans without harming regular docs.
- **OpenAI chat bridge** – `launch_chat_openai.sh` / `rag/chat_openai.py` let you run the local retrieval stack while delegating generation to any OpenAI-compatible API.
- **Platform-specific requirements** – curated `requirements*.txt` files for generic, RTX, Windows, and macOS setups keep dependency installs predictable.
- **Incremental ingestion + attachments** – hybrid retrieval runs incrementally by default, and chat/retrieve support `--attach` for ad hoc context without rebuilding indexes.

### Keeping the index up to date

- `ingest/build_index_hybrid_fast.py` is incremental by default—each run scans the `sources:` paths, calculates hashes, and only embeds new or modified chunks while pruning deletions. You can force the behavior explicitly with `--incremental` (already set in the script).
- To add content, just drop files into the directories referenced in `config.yaml` and rerun the build step. There’s no need to recreate the FAISS store from scratch unless you change chunking/embedding settings.
- Need ad hoc material without touching the index? `rag/chat.py` (and `rag/chat_openai.py`) accept `--attach <folder>` plus `--attach_only` to embed those folders on the fly for that query.

### Document conversion & OCR

- `ingest/convert_docs_to_text.py` supports a structured OCR path: add `--enable-ocr` (or `enable_ocr: true` per config task) to rasterize PDFs via Tesseract before `unstructured` chunks them. Control language packs via `--ocr-lang deu+eng` or `ocr_lang: eng+fra`. The converter first tries PyMuPDF to extract HTML (preserving links/tables) and converts it to Markdown; only image-only pages fall back to OCR. DOCX files go through `mammoth` for clean Markdown (with images replaced by short `[image omitted]` placeholders so RAG embeddings stay textual). PPTX decks use `python-pptx` (slide titles, bullet text, tables rendered as Markdown), Excel files become per-sheet Markdown tables via pandas, and standalone images (\*.png/\*.jpg/\*.tiff, etc.) are OCR’d with Tesseract so scan bundles can be ingested alongside PDFs. Need to denoise OCR output? Raise `ocr_conf_threshold` (0–1) and/or `ocr_image_conf_threshold` for scan-heavy folders; leave them at `0` to keep the raw text.
- Install platform deps before running OCR:
  - **Tesseract OCR**
    - macOS: `brew install tesseract`
    - Windows: download the official installer or use `choco install tesseract`
    - Linux: `sudo apt install tesseract-ocr` (or your distro equivalent)
    - After installation, ensure `tesseract --version` works in your terminal (PATH must include the binary)
  - **Poppler** (used by `pdf2image` when rasterizing)
    - macOS: `brew install poppler`
    - Windows: `choco install poppler` or download the zip and add its `bin` folder to `PATH`
    - Linux: `sudo apt install poppler-utils`
    - If you cannot install Poppler, the pipeline falls back to a pure-Python path using PyMuPDF + Tesseract (slower but avoids the native dependency once `PyMuPDF` is installed)
- Python deps (pytesseract, pdf2image, PyMuPDF, mammoth, opencv-python-headless, unstructured[ocr], python-magic/ python-magic-bin) are already listed in the requirements files—just pick the one for your platform.
- Conversion outputs per-document `.txt` files plus `convert_report.jsonl` so you can audit OCR, empty docs, and failures.

---

## Configure Sources (config.yaml)

Top‑level keys you’ll use most:

- `paths.index_root`: where index artifacts are stored (`store/` by default)
- `index.name`: logical name for this index; prefixes all artifacts (`<name>.faiss`, `<name>.meta.parquet`, ...)
- `sources`: list of inputs; each item supports:
  - `name`: label for this source (used in metadata)
  - `type`: tag (e.g., `repo`, `docs`, `wiki`, `external`) shown in citations and usable for weighting
  - `path`: local directory to scan
  - `include`: glob patterns to include (e.g., `"**/*.md"`)
  - `exclude`: glob patterns to exclude (e.g., binaries)
  - `treat_as`: `auto` | `code` | `prose` — affects chunking strategy
  - `max_size_mb`: per‑source size cap (optional)
- `embedding`: `model`, `batch_size`, `normalize_embeddings`
- `retrieval`:
  - `initial_k`: candidate pool size before reranking/final cut
  - `bm25.enabled`: enable lexical BM25 (needs `rank_bm25` installed)
  - `fusion_weights`: weights for `semantic`, `lexical`, `identifier`
  - `source_weights`: optional per‑source boosts (e.g., `{ docs: 1.05, external: 0.95 }`)
- `prompt` and `generation`: prompt template, input token budget, default generator model id
- `symbols`: configure header scan roots if you want identifier boosts from code
- `fetch` / `crawl`: optional fetch/crawl task descriptions
- `normalize` / `convert_docs`: optional post-processing tasks to run from config (`convert_docs` supports `glob`, `overwrite`, `enable_ocr`, `ocr_lang`, `ocr_conf_threshold` where `0` disables filtering, and `ocr_image_conf_threshold` for scan-heavy folders)

Config override
- All tools read `RAG_CONFIG` env var. Set it to use an alternate file like `config.test.yaml`.

Artifacts
- Written to `paths.index_root` with `index.name` prefix:
  - `name.faiss`, `name.meta.parquet`, `name.embeddings.npy`, `name.manifest.json`

---

## Ingestion & Tools

- Build index (incremental, CPU/GPU)
  - `python ingest/build_index_hybrid_fast.py [--gpu] [--batch N] [--threads M] [--incremental] [--limit K]`
  - Producer (CPU): scans + chunks files
  - Consumer (GPU/CPU): embeddings with OOM backoff and CPU fallback

- Crawl websites (args or config)
  - CLI: `python ingest/crawl_web.py --seeds <url1> <url2> --allow_domains example.com --depth 1 --out_dir ./data/crawl`
  - Config: `python ingest/crawl_web.py --from-config` (reads `crawl:`)
  - Respects robots.txt; saves HTML to safe paths

- Normalize HTML to text (args or config)
  - CLI: `python ingest/normalize_text.py --src ./data/crawl --dst ./data/crawl_txt`
  - Config: `python ingest/normalize_text.py --from-config` (reads `normalize:`)

- Convert PDFs/Office to text (args or config)
  - CLI: `python ingest/convert_docs_to_text.py --src ./data/docs --dst ./data/docs_txt [--glob ...] [--overwrite]`
  - Config: `python ingest/convert_docs_to_text.py --from-config` (reads `convert_docs:`)
  - OCR: add `--enable-ocr` (or `enable_ocr: true` in config) to force Tesseract-based PDF OCR; set languages via `--ocr-lang deu+eng`

- Fetch repos/pages from config
  - `python ingest/fetch_sources.py`

- Build symbol index (code headers)
  - `python rag/build_symbols.py`
  - Optional and C/C++-centric: it scans `.h`/`.inc` headers to extract identifiers used for symbol boosts. Skip this step entirely if your corpus isn’t C/C++ or you don’t need identifier boosts.

### Script overview

| Script | Purpose |
| --- | --- |
| `ingest/fetch_sources.py` | Runs the `fetch:` list from config (clone Git repos, download pages/files). |
| `ingest/crawl_web.py` | Generic crawler (CLI args or `--from-config`); saves HTML for later normalization. |
| `ingest/normalize_text.py` | Converts HTML/markup to plain text (CLI args or `--from-config`). |
| `ingest/convert_docs_to_text.py` | Converts PDF/Office/LibreOffice docs to text via `unstructured` (CLI args or `--from-config`). |
| `ingest/build_index_hybrid_fast.py` | Chunk + embed + build FAISS index (incremental, CPU/GPU, OOM fallback). |
| `rag/retrieve.py` | Hybrid retrieval (embeddings + BM25 + optional symbols + optional reranker). |
| `rag/chat.py` | Full RAG chat using a local HF generator (sessions, attachments, verbose logging). |
| `rag/chat_openai.py` | Chat using an OpenAI-compatible Chat Completions API (retrieval stays local). |
| `rag/build_symbols.py` | Optional C/C++ header scanner that powers identifier boosts; skip if not needed. |
| `tools/doctor.py` | Preflight check: packages, config, index paths, CUDA/Tensor Core detection. |
| `tools/compare_rag.py` | Runs chat twice (model-only vs RAG) and prints answers with timings. |

---

## Retrieval & Chat

- Retrieval
  - `python rag/retrieve.py --q "..." [--gpu] [--initial_k N] [--final_k M] [--use_reranker] [--reranker_id MODEL] [--show]`
  - Fuses semantic + (optional) BM25 + (optional) identifiers; optional CrossEncoder rerank

- Chat
  - `python rag/chat.py --q "..." [--gpu] [--initial_k N] [--final_k M] [--use_reranker] [--reranker_id MODEL]`
  - Generation flags: `--model_id`, `--max_new_tokens`, `--temp`
  - Sessions/memory: `--session`, `--history_k`, `--reuse_context`, `--save_context`
  - Attachments: `--attach <paths...>`, `--attach_only`, `--attach_limit_files`, `--attach_limit_chunks`
  - Debug: `--show_sources`, `--show_prompt`
  - OpenAI-compatible generation: `python rag/chat_openai.py --q "..." --final_k 3 [--use_reranker] [--gpu] [--model gpt-4o-mini] [--base_url ...] [--api_key ...]`
    - Uses `OPENAI_API_KEY` and optional `OPENAI_BASE_URL` env vars if not provided as args.

## Launch Scripts

- Normal config: `./launch.sh [--gpu|--mps]` (uses `config.yaml`)
- Test config: `./launch_test.sh [--gpu|--mps]` (uses `config.test.yaml` and seeds a tiny corpus)

Generated artifacts (safe to delete for a fresh run)
- `store/` – FAISS index + parquet + numpy + manifest (per `index.name`).
- `data/` – downloaded/cloned/crawled sources (`./data/repo`, `./data/wiki`, `./data/wiki_txt`, ...).
- `sessions/` – chat session logs (`sessions/<id>.jsonl`).
- `saved_contexts/` – saved prompts/context snapshots.
- Temporary reports/logs (e.g., `convert_report.jsonl` under `data/...`).
- If you ever run attachment-heavy tests, any scratch directories you create under `data/` can also be removed.

To reset everything, simply delete these directories:
```bash
rm -rf store data sessions saved_contexts
```
Then re-run the fetch/crawl/normalize/convert + index steps to rebuild from scratch (e.g., `./launch.sh`).

---

## Accelerator Usage

- CPU is the default. Add `--gpu` to index/retrieve/chat on NVIDIA systems (CUDA); add `--mps` on Apple Silicon (Metal). Scripts fail fast if the requested backend isn’t available.
- Tensor Core-friendly settings are enabled on CUDA paths, and MPS uses FP16 automatically. Doctor can tell you if CUDA/MPS and Tensor Cores are available.

Check your environment quickly:
- `python tools/doctor.py`

---

## Troubleshooting

- Parquet engine missing: install `pyarrow` (already in requirements)
- BM25 disabled: install `rank_bm25` (already in requirements)
- Windows HF Hub symlink warning: set `HF_HUB_DISABLE_SYMLINKS_WARNING=1`
- GPU not detected: ensure a CUDA‑enabled PyTorch wheel is installed and `nvidia-smi` works
- Attachments too big: tune `--attach_limit_files` / `--attach_limit_chunks`

---

## License

This repository is intended for your internal use. Add a license notice here if you plan to publish.
