#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chat over your RAG using an OpenAI-compatible Chat Completions API.

Keeps your ingestion + retrieval stack intact, but swaps the generator for
OpenAI (or any compatible endpoint via base_url + api_key).

Env/config:
- Reads RAG_CONFIG (config.yaml) for index paths and prompt template
- Reads OPENAI_API_KEY (or --api_key)
- Optional OPENAI_BASE_URL (or --base_url) to target compatible providers

Usage:
  python rag/chat_openai.py --q "Your question" --final_k 3 --show_sources \
    [--gpu] [--use_reranker] [--model gpt-4o-mini] [--base_url ...] [--api_key ...]
"""

from __future__ import annotations
import os, re, argparse, yaml
from pathlib import Path
import numpy as np
import pandas as pd

import faiss
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

try:
    # OpenAI python client v1+
    from openai import OpenAI, NotFoundError
except Exception:
    OpenAI = None  # handled in main
    NotFoundError = None  # type: ignore

try:
    import tiktoken  # optional, for better token accounting
    try:
        TOKEN_ENCODING = tiktoken.get_encoding("o200k_base")
    except Exception:
        TOKEN_ENCODING = None
except Exception:
    tiktoken = None  # type: ignore
    TOKEN_ENCODING = None


ap = argparse.ArgumentParser()
ap.add_argument("--q", required=True, help="User question")
ap.add_argument("--initial_k", type=int, default=24)
ap.add_argument("--final_k", type=int, default=3)
ap.add_argument("--use_reranker", action="store_true")
ap.add_argument("--reranker_id", default="BAAI/bge-reranker-base")
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--mps", action="store_true")
ap.add_argument("--model", default=None, help="OpenAI model id (e.g., gpt-4o-mini)")
ap.add_argument("--base_url", default=None, help="OpenAI-compatible base URL")
ap.add_argument("--api_key", default=None, help="API key (or set OPENAI_API_KEY)")
ap.add_argument("--max_new_tokens", type=int, default=600)
ap.add_argument("--temp", type=float, default=0.2)
ap.add_argument("--show_sources", action="store_true")
ap.add_argument("--norag", action="store_true", help="Bypass retrieval/context and send the raw question to the model")
args = ap.parse_args()


# Config
CFG_PATH = os.environ.get("RAG_CONFIG", "config.yaml")
CFG = yaml.safe_load(open(CFG_PATH, "r"))
INDEX_ROOT = Path(CFG["paths"]["index_root"])  # type: ignore
INDEX_NAME = (CFG.get("index", {}) or {}).get("name", "index")
FAISS_PATH = INDEX_ROOT / f"{INDEX_NAME}.faiss"
META_PATH  = INDEX_ROOT / f"{INDEX_NAME}.meta.parquet"
SYMS_PATH  = INDEX_ROOT / "symbols.parquet"

EMB_MODEL_ID = CFG["embedding"]["model"]
EMB_NORM     = bool(CFG["embedding"].get("normalize_embeddings", True))

PROMPT_CFG       = CFG.get("prompt", {})
MAX_INPUT_TOKENS = int(PROMPT_CFG.get("max_input_tokens", 3800))
CONTEXT_SEP      = PROMPT_CFG.get("context_separator", "\n---\n")
BASE_SYSTEM_MSG  = PROMPT_CFG.get(
    "system_message",
    "You are a concise domain expert. Use only the provided CONTEXT."
)
TEMPLATE         = PROMPT_CFG.get(
    "template",
    "SYSTEM:\n{system_message}\n\nQUESTION:\n{question}\n\nCONTEXT (CITED EXCERPTS):\n{context}\n\nANSWER:\n"
)

def count_tokens(text: str) -> int:
    if not text:
        return 0
    if TOKEN_ENCODING is not None:
        try:
            return len(TOKEN_ENCODING.encode(text))
        except Exception:
            pass
    # fall back to whitespace token approximation
    return len(text.split())

# OpenAI client
if OpenAI is None:
    raise SystemExit("openai package not installed. pip install openai")
api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("Missing OPENAI_API_KEY (or --api_key)")
client = OpenAI(api_key=api_key, base_url=(args.base_url or os.environ.get("OPENAI_BASE_URL")))
GEN_MODEL_ID = args.model or CFG.get("generation", {}).get("openai_model_id", "gpt-4o-mini")

# Load data (can be skipped via --norag)
bm25 = None
symbols = None
syms_set: set[str] = set()
embedder = None
USE_CUDA = False
USE_MPS = False
if args.norag:
    print("[chat:openai] --norag set: skipping FAISS/meta/embedder loads; responses use question only.")
    index = None
    meta = pd.DataFrame()
else:
    print(f"[chat:openai] loading FAISS index: {FAISS_PATH}")
    index = faiss.read_index(str(FAISS_PATH))
    meta  = pd.read_parquet(META_PATH)
    print(f"[chat:openai] meta rows: {len(meta)}")

    # Optional BM25
    try:
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi([t.split() for t in meta["text"].astype(str)])
    except Exception:
        bm25 = None

    # Optional symbols
    if SYMS_PATH.exists():
        symbols = pd.read_parquet(SYMS_PATH)
        syms_set = set(symbols["symbol"].unique())

    # Embedder (retrieval)
    print(f"[chat:openai] loading embedder: {EMB_MODEL_ID}")
    embedder = SentenceTransformer(EMB_MODEL_ID)
    HAS_MPS = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    if args.gpu and args.mps:
        raise SystemExit("--gpu and --mps cannot be used together.")
    if args.gpu and not torch.cuda.is_available():
        raise SystemExit("--gpu set but CUDA not available.")
    if args.mps and not HAS_MPS:
        raise SystemExit("--mps set but PyTorch MPS backend is unavailable.")
    USE_CUDA = bool(args.gpu and torch.cuda.is_available())
    USE_MPS = bool(args.mps and HAS_MPS)
    if USE_CUDA:
        try:
            embedder = embedder.to("cuda")
            first = getattr(embedder, "_first_module", None)
            if callable(first):
                mod = first()
                auto_model = getattr(mod, "auto_model", None)
                if auto_model is not None:
                    auto_model.half()
        except Exception:
            print("[chat:openai] failed to move embedder to CUDA, falling back to CPU")
            embedder = embedder.to("cpu")
            USE_CUDA = False
    elif USE_MPS:
        try:
            embedder = embedder.to("mps")
        except Exception:
            print("[chat:openai] failed to move embedder to MPS, falling back to CPU")
            embedder = embedder.to("cpu")
            USE_MPS = False


# Retrieval utils (mirrors rag/retrieve.py)
def semantic_search(q: str, k: int) -> pd.DataFrame:
    if USE_CUDA:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            qvec = embedder.encode([q], batch_size=1, convert_to_tensor=True, normalize_embeddings=EMB_NORM, show_progress_bar=False)
        qv = qvec.detach().float().cpu().numpy()
    elif USE_MPS:
        qvec = embedder.encode([q], batch_size=1, convert_to_tensor=True, normalize_embeddings=EMB_NORM, show_progress_bar=False)
        qv = qvec.detach().cpu().numpy()
    else:
        qv = embedder.encode([q], normalize_embeddings=EMB_NORM)
    scores, ids = index.search(np.asarray(qv, dtype="float32"), k)
    rows = meta.iloc[ids[0]].copy()
    rows["sem_score"] = scores[0]
    return rows

def lexical_search(q: str, k: int) -> pd.DataFrame:
    if not bm25:
        return pd.DataFrame()
    scores = bm25.get_scores(q.split())
    out = meta.copy()
    out["lex_score"] = scores
    return out.nlargest(k, "lex_score")

def identifier_hits(q: str) -> pd.DataFrame:
    if symbols is None:
        return pd.DataFrame()
    # Simple identifier extraction (letters/digits/underscore)
    toks = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\b", q)
    if not toks:
        return pd.DataFrame()
    hit = symbols[symbols["symbol"].isin(toks)]
    m = meta
    out_rows = []
    for _, r in hit.iterrows():
        cand = m[m["path"] == r["path"]]
        if cand.empty: continue
        sl, el = int(r["start_line"]), int(r["end_line"]) 
        c = cand
        if "start_line" in c.columns and "end_line" in c.columns:
            c = c[(pd.notna(c["start_line"])) & (pd.notna(c["end_line"]))]
            try:
                c = c[(c["start_line"].astype(int) <= el) & (c["end_line"].astype(int) >= sl)]
            except Exception:
                pass
        if c.empty:
            c = cand.head(3)
        if not c.empty:
            c = c.copy(); c["id_score"] = 1.0; out_rows.append(c)
    if not out_rows:
        return pd.DataFrame()
    return pd.concat(out_rows, ignore_index=True).drop_duplicates("id")

def fuse(q: str, k_init: int, k_final: int) -> pd.DataFrame:
    sem = semantic_search(q, k_init)
    lex = lexical_search(q, k_init)
    ide = identifier_hits(q)
    all_ = pd.concat([sem, lex, ide], ignore_index=True).drop_duplicates("id")
    # normalize scores 0..1
    for col in ("sem_score","lex_score","id_score"):
        if col not in all_.columns:
            all_[col] = 0.0
        else:
            s = all_[col].astype(float)
            if s.max() > 0:
                all_[col] = (s - s.min()) / (s.max() - s.min() + 1e-9)
            else:
                all_[col] = 0.0
    fw = (CFG.get("retrieval", {}) or {}).get("fusion_weights", {}) or {}
    w_sem = float(fw.get("semantic", 0.55))
    w_lex = float(fw.get("lexical", 0.35))
    w_id  = float(fw.get("identifier", 0.10))
    all_["fused"] = w_sem*all_["sem_score"] + w_lex*all_["lex_score"] + w_id*all_["id_score"]
    return all_.nlargest(max(k_init, k_final), "fused")

def rerank(q: str, df: pd.DataFrame, k_final: int, model_id: str) -> pd.DataFrame:
    if df.empty or len(df) <= k_final:
        return df.head(k_final)
    rr = CrossEncoder(model_id)
    pairs = [(q, str(t)[:3000]) for t in df["text"].astype(str)]
    scores = rr.predict(pairs)
    df = df.copy(); df["rr_score"] = scores
    return df.nlargest(k_final, "rr_score")


# Prompt helpers
def fmt_citation(row) -> str:
    st = row.get("source_type", "")
    path = str(row.get("path", "") or row.get("url", ""))
    sl, el = row.get("start_line"), row.get("end_line")
    if sl is not None and pd.notna(sl) and el is not None and pd.notna(el):
        return f"[{st}:{path} L{int(sl)}-{int(el)}]"
    return f"[{st}:{path}]"

def render_context_sections(rows: pd.DataFrame) -> list[str]:
    blocks = []
    for _, r in rows.iterrows():
        cite = fmt_citation(r)
        txt = str(r["text"]).strip()
        blocks.append(f"{cite}\n{txt}")
    return blocks


def main():
    q = args.q.strip()
    if args.norag:
        pool = pd.DataFrame()
        top = pd.DataFrame()
        ctx_blocks: list[str] = []
    else:
        pool = fuse(q, args.initial_k, args.final_k)
        top = rerank(q, pool, args.final_k, args.reranker_id) if args.use_reranker else pool.head(args.final_k)
        ctx_blocks = render_context_sections(top)
    ctx_text = CONTEXT_SEP.join(ctx_blocks)

    system_message = BASE_SYSTEM_MSG
    user_message = TEMPLATE.format(system_message=system_message, question=q, context=ctx_text)
    system_tokens = count_tokens(system_message)
    question_tokens = count_tokens(q)
    context_tokens = count_tokens(ctx_text)
    prompt_payload_tokens = count_tokens(user_message)

    base_url_obj = getattr(client, "base_url", "openai")
    base_url_str = str(base_url_obj or "openai")
    print(f"[chat:openai] calling model={GEN_MODEL_ID} base_url={base_url_str}")
    base_url_obj = getattr(client, "base_url", "openai")
    base_url_str = str(base_url_obj)
    endpoint_hint = f"{base_url_str.rstrip('/')}/chat/completions"
    try:
        resp = client.chat.completions.create(
            model=GEN_MODEL_ID,
            messages=[
                {"role":"system","content": system_message},
                {"role":"user","content": user_message}
            ],
            temperature=float(args.temp),
            max_tokens=int(args.max_new_tokens),
        )
    except Exception as err:
        if NotFoundError and isinstance(err, NotFoundError):
            print(f"[chat:openai] 404 from {endpoint_hint} (model={GEN_MODEL_ID})")
        raise
    answer = resp.choices[0].message.content or ""
    if "ANSWER:" in answer:
        answer = answer.split("ANSWER:", 1)[-1].strip()
    print("\n" + answer + "\n")

    usage = getattr(resp, "usage", None)
    if usage is not None:
        prompt_tokens_api = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
        completion_tokens_api = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
        total_tokens_api = getattr(usage, "total_tokens", None)
        print("[chat:openai] token usage (API): "
              f"prompt={prompt_tokens_api} completion={completion_tokens_api} total={total_tokens_api}")
    response_tokens_est = count_tokens(answer)
    print("[chat:openai] token estimate (local): "
          f"system={system_tokens} question={question_tokens} context={context_tokens} "
          f"prompt_payload={prompt_payload_tokens} response~{response_tokens_est} "
          f"total~{system_tokens + prompt_payload_tokens + response_tokens_est}")

    if args.show_sources and not top.empty:
        print("Sources:")
        for _, r in top.iterrows():
            cite = fmt_citation(r)
            path = r.get("path", "") or r.get("url", "")
            sl, el = r.get("start_line"), r.get("end_line")
            loc = f" L{int(sl)}-{int(el)}" if pd.notna(sl) and pd.notna(el) else ""
            fused = r.get("fused", 0.0)
            rr    = r.get("rr_score", 0.0)
            print(f" - {cite}  {path}{loc}  (fused={fused:.3f}{', rr='+format(rr,'.3f') if rr else ''})")


if __name__ == "__main__":
    main()
