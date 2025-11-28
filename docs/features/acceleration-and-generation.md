# Feature: Accelerator-Aware Execution and Generation Bridge

## Overview
Run on CPU by default while enabling CUDA or Apple MPS acceleration, and optionally delegate generation to an OpenAI-compatible API while keeping local retrieval.

## Goals
- Provide predictable CPU defaults that work everywhere.
- Let users opt into GPU/MPS for faster embedding and generation when available.
- Offer an OpenAI bridge without changing retrieval code.

## User stories
- As an operator, I add `--gpu` to indexing and chat commands to speed up runs on NVIDIA hardware.
- As a Mac user, I add `--mps` to leverage Apple Metal for embedding.
- As a user, I call `rag/chat_openai.py --q ... --final_k 3 --model gpt-4o-mini` to use an external generator with local retrieval.

## Requirements
- CLI flags `--gpu` and `--mps` exist on index/retrieve/chat scripts; they fail fast if the requested backend is unavailable and fall back to CPU when possible.
- CUDA path halves transformer weights when possible; MPS path uses the Apple backend.
- Generation model can be set via config (`generation.model_id`) or `--model_id`; OpenAI bridge uses `OPENAI_API_KEY` and optional `OPENAI_BASE_URL`.
- Launch scripts (`launch.sh`, `launch_test.sh`, `launch_chat_openai.sh`) surface the same flags for common flows.

## Flow
1. User selects device via CLI flags; scripts validate availability.
2. Embedding and (for chat) generation models load on the chosen device; otherwise fall back to CPU with warnings.
3. Retrieval executes as normal; prompt assembly is unchanged.
4. For OpenAI bridge, retrieval stays local and the final prompt is sent to the external API for completion.

## Acceptance criteria
- CPU path works without CUDA/MPS installed.
- Requesting unavailable accelerators yields clear errors without partial failures.
- OpenAI bridge returns answers that include local citations; failure to reach the API is surfaced without corrupting local retrieval.

## Risks / open questions
- GPU/driver mismatches can break acceleration; keep requirements files aligned with supported wheels.
- External API usage depends on network access and quotas; provide fallbacks to local generation.

## Non-functional targets and validation
- Device performance: GPU/MPS embedding throughput should be >2x CPU on supported hardware; log device and batch size.
- Latency: chat with reranker on CPU p95 <4s on moderate index; GPU/MPS improves to <2.5s where available.
- Failure modes: unavailable device must fail fast; no silent fallback without a warning.

## Error handling and logging
- Device selection errors (CUDA init, MPS unsupported) emit clear messages and either fall back (with warning) or exit per flag.
- OpenAI bridge: network/auth failures surfaced with status/code; no retries with other keys.
- Log whether CPU/GPU/MPS was used for embedding/generation and whether fallback occurred.

## Security/privacy/ops
- API keys for OpenAI bridge come from env/CLI only; never written to disk. Recommend rotation and scoped keys.
- No request logging of prompt text beyond optional debug; redact secrets from logs.
- Document driver/torch wheel requirements per platform in requirements files; pin versions.

## Eval checklist
- Run chat/retrieve on CPU, then with `--gpu`/`--mps`; compare throughput and log device selection.
- Simulate missing CUDA/MPS to confirm clear failure or warned fallback.
- Use OpenAI bridge with and without valid key/base URL to validate auth error handling and local retrieval preservation.
