#!/usr/bin/env bash
set -euo pipefail

# Generic end-to-end smoke test for the config-driven RAG
#
# Steps:
#  1) (Optional) install deps
#  2) Create a tiny local corpus in ./data/repo
#  3) (Optional) fetch/crawl sources from config snippets
#  4) (Optional) normalize HTML to text / convert Office/PDF to text
#  5) Build the index (GPU default; use --cpu to force CPU)
#  6) Run a retrieval sanity check
#  7) (Optional) run chat

echo "[1/7] (Optional) Install dependencies"
echo "      pip install -r requirements.txt"
echo "      # Preflight: RAG_CONFIG=config.test.yaml python tools/doctor.py"

echo "[2/7] Create tiny sample corpus under ./data/repo"
mkdir -p ./data/repo
cat > ./data/repo/math.c << 'EOF'
int add(int a,int b){return a+b;}
int mul(int a,int b){return a*b;}
EOF
cat > ./data/repo/README.md << 'EOF'
# Tutorial: Using add and mul
Call add(2,3) to get 5. mul(2,3) returns 6.
EOF

echo "[3/7] (Optional) Fetch or crawl sources from config"
echo "      py ingest/fetch_sources.py"
echo "      py ingest/crawl_web.py --seeds https://example.com/docs --allow_domains example.com --depth 1 --out_dir ./data/crawl"

echo "[4/7] (Optional) Normalize HTML to text / convert Office/PDF"
echo "      py ingest/normalize_text.py --src ./data/crawl --dst ./data/crawl_txt"
echo "      py ingest/convert_docs_to_text.py --src ./data/docs --dst ./data/docs_txt"

# ------------------------------------------------------------
# Example: End-to-end Docs pipeline (PDF/Office -> text -> index)
# ------------------------------------------------------------
# 1) Put your PDFs/Office files into ./data/docs (or fetch/crawl to a folder)
#    mkdir -p ./data/docs
#    # copy .pdf/.docx/.pptx/.xlsx/.odt, etc. into ./data/docs
#
# 2) Convert to text while preserving structure
#    py ingest/convert_docs_to_text.py --src ./data/docs --dst ./data/docs_txt
#
# 3) Ensure config.yaml has the Docs source (already included):
#    - name: Docs
#      type: docs
#      path: ./data/docs_txt
#      include: ["**/*.txt"]
#      treat_as: prose
#
# 4) Rebuild index (incremental is fine)
#    py ingest/build_index_hybrid_fast.py --incremental --batch 32
#
# 5) Ask questions grounded in those docs
#    py rag/chat.py --q "Summarize the onboarding policy" --final_k 3 --show_sources

ACCEL_FLAG=""; BATCH=8
if [[ "${1:-}" == "--gpu" ]]; then ACCEL_FLAG="--gpu"; BATCH=64
elif [[ "${1:-}" == "--mps" ]]; then ACCEL_FLAG="--mps"; BATCH=32
fi

echo "[5/7] Build index ($([[ -n "$ACCEL_FLAG" ]] && echo accel || echo CPU)), batch=$BATCH"
RAG_CONFIG=${RAG_CONFIG:-config.yaml} python ingest/build_index_hybrid_fast.py $ACCEL_FLAG --batch $BATCH || {
  echo "Index build failed"; exit 1;
}

echo "[6/7] Retrieval sanity check"
RAG_CONFIG=${RAG_CONFIG:-config.yaml} python rag/retrieve.py $ACCEL_FLAG --q "How to use add and mul functions?" --show || {
  echo "Retrieval failed"; exit 1;
}

echo "[7/7] (Optional) Chat over your corpus"
ACCEL_CHAT=""
if [[ "$ACCEL_FLAG" == "--gpu" ]]; then ACCEL_CHAT="--gpu"
elif [[ "$ACCEL_FLAG" == "--mps" ]]; then ACCEL_CHAT="--mps"
fi
echo "      RAG_CONFIG=config.test.yaml python rag/chat.py $ACCEL_CHAT --q \"Show a snippet calling add(2,3) and mul(2,3)\" --final_k 3 --show_sources"

# CPU-only example (force CPU and smaller batch)
#   RAG_CONFIG=config.test.yaml python ingest/build_index_hybrid_fast.py --cpu --batch 8

# Attachments-only example (no rebuild; GPU encodes attachments by default)
#   RAG_CONFIG=config.test.yaml python rag/chat.py \
#     --q "Summarize these docs" \
#     --attach ./data/docs \
#     --attach_only --attach_limit_files 100 --attach_limit_chunks 1000 \
#     --final_k 5 --show_sources

echo "Done. Edit config.yaml 'sources:' to point at your real data, then rerun steps 5-7."
