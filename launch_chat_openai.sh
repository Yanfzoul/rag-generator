#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper around rag/chat_openai.py that lets you set/OpenAI env vars
# alongside common knobs (config, model, accel flags) in one place.

usage() {
  cat <<'EOF'
Usage: ./launch_chat_openai.sh --q "What is the onboarding process?" [options]

Required:
  --q TEXT               Question to send to the RAG stack (quotes recommended)

Optional helpers:
  --config PATH          Override config file (defaults to RAG_CONFIG or config.yaml)
  --api-key KEY          Provide OPENAI_API_KEY inline (else env/interactive prompt)
  --base-url URL         Override OPENAI_BASE_URL (for Azure / compat endpoints)
  --model ID             Override generation model (e.g., gpt-4o-mini)
  --norag                Forward to chat_openai.py to skip retrieval context (raw LLM baseline)
  --env-prompt           Prompt for OPENAI_API_KEY if still unset

Everything else (e.g., --gpu, --mps, --final_k, --use_reranker, --show_sources)
is passed straight through to rag/chat_openai.py.

Examples:
  ./launch_chat_openai.sh --q "Summarize release notes" --show_sources
  ./launch_chat_openai.sh --q "List APIs" --gpu --model gpt-4o-mini \
    --api-key sk-... --base-url https://api.openai.com/v1
EOF
}

QUESTION=""
CONFIG_PATH="${RAG_CONFIG:-config.sgdk.yaml}"
API_KEY_ARG=""
BASE_URL_ARG=""
MODEL_ARG=""
PROMPT_FOR_KEY="false"
PY_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --q|--question)
      QUESTION="$2"; shift 2 ;;
    --config)
      CONFIG_PATH="$2"; shift 2 ;;
    --api-key)
      API_KEY_ARG="$2"; shift 2 ;;
    --base-url)
      BASE_URL_ARG="$2"; shift 2 ;;
    --model)
      MODEL_ARG="$2"; shift 2 ;;
    --env-prompt)
      PROMPT_FOR_KEY="true"; shift ;;
    --help|-h)
      usage; exit 0 ;;
    *)
      PY_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$QUESTION" ]]; then
  echo "error: --q is required" >&2
  usage
  exit 1
fi

export RAG_CONFIG="$CONFIG_PATH"

if [[ -n "$API_KEY_ARG" ]]; then
  export OPENAI_API_KEY="$API_KEY_ARG"
fi
if [[ -n "$BASE_URL_ARG" ]]; then
  export OPENAI_BASE_URL="$BASE_URL_ARG"
fi

if [[ -z "${OPENAI_API_KEY:-}" ]] && [[ "$PROMPT_FOR_KEY" == "true" ]]; then
  read -r -s -p "Enter OPENAI_API_KEY: " READ_KEY
  echo
  if [[ -n "$READ_KEY" ]]; then
    export OPENAI_API_KEY="$READ_KEY"
  fi
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "error: OPENAI_API_KEY must be set (env var, --api-key, or --env-prompt)" >&2
  exit 1
fi

CMD=(python rag/chat_openai.py --q "$QUESTION")
if [[ -n "$MODEL_ARG" ]]; then
  CMD+=(--model "$MODEL_ARG")
fi
CMD+=("${PY_ARGS[@]}")

echo "[chat-openai] RAG_CONFIG=$RAG_CONFIG"
echo "[chat-openai] OPENAI_BASE_URL=${OPENAI_BASE_URL:-openai-default}"
echo "[chat-openai] model=${MODEL_ARG:-'(default from config)'}"
echo "[chat-openai] extras: ${PY_ARGS[*]:-"(none)"}"
echo

"${CMD[@]}"
