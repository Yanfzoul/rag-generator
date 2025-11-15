#!/usr/bin/env bash
set -euo pipefail

# Launch: Test case with config.test.yaml (small corpus)
# Usage: ./launch_test.sh [--gpu]

ACCEL_FLAG=""; BATCH=8
if [[ "${1:-}" == "--gpu" ]]; then ACCEL_FLAG="--gpu"; BATCH=64
elif [[ "${1:-}" == "--mps" ]]; then ACCEL_FLAG="--mps"; BATCH=32
fi

export RAG_CONFIG=config.test.yaml

echo "[launch:test] Using config: $RAG_CONFIG"
echo "[launch:test] Preflight (doctor)"
python tools/doctor.py || true

echo "[launch:test] Seed tiny sample corpus under ./data/repo"
mkdir -p ./data/repo
cat > ./data/repo/math.c << 'EOF'
int add(int a,int b){return a+b;}
int mul(int a,int b){return a*b;}
EOF
cat > ./data/repo/README.md << 'EOF'
# Tutorial: Using add and mul
Call add(2,3) to get 5. mul(2,3) returns 6.
EOF

echo "[launch:test] Build index ($([[ -n "$ACCEL_FLAG" ]] && echo accel || echo CPU)), batch=$BATCH"
python ingest/build_index_hybrid_fast.py $ACCEL_FLAG --batch $BATCH

echo "[launch:test] Retrieval sanity check"
python rag/retrieve.py $ACCEL_FLAG --q "How to use add and mul functions?" --show

echo "[launch:test] Chat quick check"
python rag/chat.py $ACCEL_FLAG --q "Show a snippet calling add(2,3) and mul(2,3)" --final_k 3 --show_sources

ACCEL_COMPARE=""
if [[ "$ACCEL_FLAG" == "--gpu" ]]; then ACCEL_COMPARE="--gpu"
elif [[ "$ACCEL_FLAG" == "--mps" ]]; then ACCEL_COMPARE="--mps"
fi
echo "[launch:test] Compare baseline vs RAG"
python tools/compare_rag.py $ACCEL_COMPARE --q "How to init SGDK and draw a sprite?" --rag_final_k 3 --show_sources

echo "[launch:test] Done."
