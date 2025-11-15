# Setup Guide (Linux, macOS, GitLab CI)

This guide covers a venv‑optional installation and usage flow on Linux/macOS and GitLab runners. CUDA is optional; the project runs fully on CPU.

## 1) Install Python packages (no venv required)

Linux/macOS (system Python)
- python3 -m pip install --upgrade pip
- python3 -m pip install -r requirements.txt
- Optional lexical retriever: python3 -m pip install rank_bm25

Optional venv (if you prefer isolation)
- python3 -m venv .venv && source .venv/bin/activate
- pip install --upgrade pip
- pip install -r requirements.txt

## 2) Optional: GPU on Linux (CUDA)

Install a CUDA‑enabled PyTorch matching your driver version (examples):
- CUDA 11.8: python3 -m pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision
- CUDA 12.1: python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision

Verify CUDA:
- python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.version.cuda)"

Notes
- You do not need the full CUDA Toolkit; the PyTorch wheel bundles CUDA.
- If CUDA is not available, run on CPU (omit --gpu flags).

## 2b) Windows (Git Bash)

Use the same commands as Linux/macOS inside Git Bash. Tips:

- Install deps (no venv):
  - `python -m pip install --upgrade pip`
  - `python -m pip install -r requirements.txt`
  - Optional: `python -m pip install rank_bm25`

- Optional venv (Git Bash):
  - `python -m venv .venv && source .venv/Scripts/activate`

- Silence HF Hub symlink warnings on Windows:
  - `export HF_HUB_DISABLE_SYMLINKS_WARNING=1`

- CUDA wheels on Windows (if GPU available):
  - CUDA 11.8: `python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision`
  - CUDA 12.1: `python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision`
  - Verify: `python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.version.cuda)"`
  - If torchaudio fails to install on your combo, omit it — not required by this project.

- RTX 2080 quick path:
  - `python -m pip install --upgrade pip`
  - `python -m pip install -r requirements_rtx2080.txt`

## 2c) Apple Silicon (macOS MPS)

- Use the same steps as Section 1 (system Python or venv).
- Recent PyTorch wheels on macOS include Metal (MPS) support by default; no CUDA toolkit required.
- When running scripts, add `--mps` instead of `--gpu` to offload retrieval/chat/index computations to the GPU:
  - `python ingest/build_index_hybrid_fast.py --mps --batch 32`
  - `python rag/retrieve.py --mps --q "How to use add and mul functions?" --show`
  - `python rag/chat.py --mps --q "Show a snippet calling add(2,3) and mul(2,3)" --final_k 3 --show_sources`
- Leave off `--mps` to fall back to CPU at any time.

## 3) Choose a config and run the doctor

- Normal config: export RAG_CONFIG=config.yaml
- Test config:   export RAG_CONFIG=config.test.yaml
- SGDK config:   export RAG_CONFIG=config.sgdk.yaml

Preflight:
- python3 tools/doctor.py

## 4) Quick start (CPU by default)

Build index
- python3 ingest/build_index_hybrid_fast.py --batch 8

Retrieve and chat
- python3 rag/retrieve.py --q "How to use add and mul functions?" --show
- python3 rag/chat.py --q "Show a snippet calling add(2,3) and mul(2,3)" --final_k 3 --show_sources

Compare baseline vs RAG
- python3 tools/compare_rag.py --q "How to init SGDK and draw a sprite?" --rag_final_k 3 --show_sources

GPU variants (if CUDA available)
- Add --gpu to index/retrieve/chat/compare and increase batch, e.g.:
  - python3 ingest/build_index_hybrid_fast.py --gpu --batch 64

## 5) Config‑driven web‑first pipeline (SGDK)

Select SGDK config
- export RAG_CONFIG=config.sgdk.yaml

Fetch/crawl/normalize/convert from config
- python3 ingest/fetch_sources.py
- python3 ingest/crawl_web.py --from-config
- python3 ingest/normalize_text.py --from-config
- python3 ingest/convert_docs_to_text.py --from-config   # if convert_docs: is set

Optional symbols
- python3 rag/build_symbols.py

Build index and use
- python3 ingest/build_index_hybrid_fast.py --batch 8      # or --gpu --batch 64
- python3 rag/retrieve.py --q "Play PCM samples with XGM" --show
- python3 rag/chat.py --q "How to init SGDK and draw a sprite?" --final_k 3 --show_sources

## 6) Launch scripts (convenience)

- Normal: ./launch.sh [--gpu]           (uses config.yaml)
- Test:   ./launch_test.sh [--gpu]      (uses config.test.yaml and seeds tiny corpus)
- SGDK:   ./launch_sgdk.sh [--gpu]      (uses config.sgdk.yaml; runs fetch/crawl/normalize/convert)

## 7) GitLab CI (CPU example)

A minimal CI pipeline that installs deps, builds an index (test config), and runs a retrieval check.

```yaml
image: python:3.11

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.pip-cache"
  RAG_CONFIG: "config.test.yaml"

cache:
  key: pip-cache
  paths:
    - .pip-cache

stages: [setup, build, retrieve]

setup:
  stage: setup
  script:
    - python -m pip install --upgrade pip
    - pip install -r requirements.txt
  artifacts:
    paths:
      - store
      - data
    expire_in: 1 week

build:
  stage: build
  script:
    - python tools/doctor.py || true
    - python ingest/build_index_hybrid_fast.py --batch 8

retrieve:
  stage: retrieve
  script:
    - python rag/retrieve.py --q "How to use add and mul functions?" --show
```

GPU runners (optional)
- Use a Docker image with NVIDIA runtime and install CUDA wheels for PyTorch in the setup job.
- Add --gpu to the build/retrieve steps and set a larger batch for the indexer.

## 8) Troubleshooting

- Parquet engine missing: pip install pyarrow (already in requirements)
- BM25 disabled: pip install rank_bm25 (already in requirements)
- CUDA not detected: ensure a CUDA‑enabled PyTorch is installed and the runner has NVIDIA drivers (check `nvidia-smi`)
- Windows symlink warning on HF Hub (for Git Bash on Windows): export HF_HUB_DISABLE_SYMLINKS_WARNING=1

## 9) Maintenance

- One script per task; each supports arguments and/or `--from-config`:
  - Crawl: ingest/crawl_web.py [--from-config]
  - Normalize: ingest/normalize_text.py [--from-config]
  - Convert: ingest/convert_docs_to_text.py [--from-config]
  - Fetch: ingest/fetch_sources.py (config‑driven)
  - Symbols: rag/build_symbols.py (config‑driven)
  - Index: ingest/build_index_hybrid_fast.py (config‑driven + runtime knobs)
  - Retrieve/Chat: rag/retrieve.py, rag/chat.py (runtime knobs; read index/generation settings from config)
